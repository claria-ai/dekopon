use std::{
    future::Future,
    num::{NonZeroU32, NonZeroUsize},
    ops::ControlFlow,
};

use aws_smithy_types::Document;
use tokio::sync::watch;

use crate::{
    bedrock::{BedrockClient, BedrockError, BedrockRequest, BedrockResponse},
    sdk::types::{
        ContentBlock, ConversationRole, Message, ReasoningContentBlock, StopReason, Tool,
        ToolResultBlock, ToolResultContentBlock, ToolUseBlock,
    },
};

pub trait InferenceModel: Send + Sync {
    fn generate<'a>(
        &'a self,
        request: BedrockRequest<'a>,
        observe: &'a mut (dyn FnMut(&str) -> ControlFlow<()> + Send),
    ) -> impl Future<Output = Result<BedrockResponse, BedrockError>> + Send + 'a;
}

impl InferenceModel for BedrockClient {
    async fn generate<'a>(
        &'a self,
        request: BedrockRequest<'a>,
        observe: &'a mut (dyn FnMut(&str) -> ControlFlow<()> + Send),
    ) -> Result<BedrockResponse, BedrockError> {
        self.converse(request, observe).await
    }
}

pub trait ToolHost: Send {
    type Error: std::error::Error + Send + Sync + 'static;

    fn execute<'a>(
        &'a mut self,
        call: &'a ToolUseBlock,
    ) -> impl Future<Output = Result<ToolResultBlock, Self::Error>> + Send + 'a;
}

#[derive(Clone, Copy)]
pub struct Limits {
    pub model_calls: NonZeroU32,
    pub tool_calls: NonZeroU32,
    pub messages: NonZeroUsize,
    /// Bounds message content bytes; system instructions, schemas and token accounting belong to the host.
    pub transcript_bytes: NonZeroUsize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Resource {
    ModelCalls,
    ToolCalls,
    Messages,
    TranscriptBytes,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProtocolFailure {
    AssistantRole,
    StopToolMismatch,
    ToolNotOffered,
    DuplicateToolId,
    ToolResultId,
    UnsupportedContent,
}

#[derive(Debug, thiserror::Error)]
pub enum AgentError<E: std::error::Error + 'static> {
    #[error("inference failed")]
    Model(#[source] BedrockError),
    #[error("tool host failed")]
    Host(#[source] E),
    #[error("tool results could not form a protocol message")]
    Message(#[source] aws_smithy_types::error::operation::BuildError),
    #[error("session cancelled")]
    Cancelled,
    #[error("session limit reached: {0:?}")]
    Limit(Resource),
    #[error("invalid model protocol: {0:?}")]
    Protocol(ProtocolFailure),
    #[error("model stopped before completion")]
    Stopped(StopReason),
}

pub struct Outcome {
    pub transcript: Vec<Message>,
    pub response: BedrockResponse,
    pub model_calls: u32,
    pub tool_calls: u32,
}

pub async fn run<M: InferenceModel, H: ToolHost>(
    model: &M,
    host: &mut H,
    mut request: BedrockRequest<'_>,
    limits: Limits,
    mut cancel: watch::Receiver<bool>,
    observe: &mut (dyn FnMut(&str) -> ControlFlow<()> + Send),
) -> Result<Outcome, AgentError<H::Error>> {
    let mut model_calls = 0;
    let mut tool_calls = 0_u32;
    loop {
        check_cancelled(&cancel)?;
        check_transcript(&request.messages, limits)?;
        if model_calls == limits.model_calls.get() {
            return Err(AgentError::Limit(Resource::ModelCalls));
        }
        let response = tokio::select! {
            biased;
            () = cancelled(&mut cancel) => return Err(AgentError::Cancelled),
            response = model.generate(request.clone(), observe) => response.map_err(AgentError::Model)?,
        };
        model_calls += 1;
        if response.message.role() != &ConversationRole::Assistant {
            return Err(AgentError::Protocol(ProtocolFailure::AssistantRole));
        }
        let calls: Vec<_> = response
            .message
            .content()
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolUse(call) => Some(call),
                _ => None,
            })
            .collect();
        let call_count = u32::try_from(calls.len()).unwrap_or(u32::MAX);
        tracing::info!(
            model_calls,
            tool_calls = call_count,
            latency_ms = response.latency_ms,
            "embedded model turn completed"
        );
        match (&response.stop_reason, calls.is_empty()) {
            (StopReason::EndTurn | StopReason::StopSequence, true) => {
                request.messages.push(response.message.clone());
                check_transcript(&request.messages, limits)?;
                check_cancelled(&cancel)?;
                return Ok(Outcome {
                    transcript: request.messages,
                    response,
                    model_calls,
                    tool_calls,
                });
            }
            (StopReason::ToolUse, false) | (StopReason::MaxTokens, false) => {}
            (StopReason::ToolUse, true)
            | (StopReason::EndTurn | StopReason::StopSequence, false) => {
                return Err(AgentError::Protocol(ProtocolFailure::StopToolMismatch));
            }
            _ => return Err(AgentError::Stopped(response.stop_reason)),
        }
        if call_count > limits.tool_calls.get().saturating_sub(tool_calls) {
            return Err(AgentError::Limit(Resource::ToolCalls));
        }
        for (index, call) in calls.iter().enumerate() {
            if calls[..index]
                .iter()
                .any(|prior| prior.tool_use_id() == call.tool_use_id())
                || request
                    .messages
                    .iter()
                    .flat_map(|message| message.content())
                    .any(|block| match block {
                        ContentBlock::ToolUse(prior) => prior.tool_use_id() == call.tool_use_id(),
                        _ => false,
                    })
            {
                return Err(AgentError::Protocol(ProtocolFailure::DuplicateToolId));
            }
            let offered = request.tools.as_ref().is_some_and(|config| {
                config.tools().iter().any(|tool| match tool {
                    Tool::ToolSpec(spec) => spec.name() == call.name(),
                    _ => false,
                })
            });
            if !offered {
                return Err(AgentError::Protocol(ProtocolFailure::ToolNotOffered));
            }
        }
        if request.messages.len().saturating_add(2) > limits.messages.get() {
            return Err(AgentError::Limit(Resource::Messages));
        }
        request.messages.push(response.message.clone());
        check_transcript(&request.messages, limits)?;
        let mut results = Vec::with_capacity(calls.len());
        for call in calls {
            check_cancelled(&cancel)?;
            // An admitted host effect owns its commit; cancellation only prevents the next effect.
            let result = host.execute(call).await.map_err(AgentError::Host)?;
            if result.tool_use_id() != call.tool_use_id() {
                return Err(AgentError::Protocol(ProtocolFailure::ToolResultId));
            }
            results.push(ContentBlock::ToolResult(result));
            tool_calls += 1;
        }
        let results = Message::builder()
            .role(ConversationRole::User)
            .set_content(Some(results))
            .build()
            .map_err(AgentError::Message)?;
        request.messages.push(results);
    }
}

fn check_cancelled<E: std::error::Error>(
    cancel: &watch::Receiver<bool>,
) -> Result<(), AgentError<E>> {
    if *cancel.borrow() {
        Err(AgentError::Cancelled)
    } else {
        Ok(())
    }
}

async fn cancelled(cancel: &mut watch::Receiver<bool>) {
    loop {
        if *cancel.borrow_and_update() {
            return;
        }
        if cancel.changed().await.is_err() {
            std::future::pending::<()>().await;
        }
    }
}

fn check_transcript<E: std::error::Error>(
    messages: &[Message],
    limits: Limits,
) -> Result<(), AgentError<E>> {
    if messages.len() > limits.messages.get() {
        return Err(AgentError::Limit(Resource::Messages));
    }
    let mut bytes = 0_usize;
    for message in messages {
        for block in message.content() {
            let count = match block {
                ContentBlock::Text(text) => text.len(),
                ContentBlock::CachePoint(_) => 0,
                ContentBlock::ToolUse(call) => call
                    .tool_use_id()
                    .len()
                    .saturating_add(call.name().len())
                    .saturating_add(document_bytes(call.input())),
                ContentBlock::ToolResult(result) => {
                    let mut total = result.tool_use_id().len();
                    for content in result.content() {
                        total = total.saturating_add(match content {
                            ToolResultContentBlock::Text(text) => text.len(),
                            ToolResultContentBlock::Json(document) => document_bytes(document),
                            _ => {
                                return Err(AgentError::Protocol(
                                    ProtocolFailure::UnsupportedContent,
                                ));
                            }
                        });
                    }
                    total
                }
                ContentBlock::ReasoningContent(reasoning) => match reasoning {
                    ReasoningContentBlock::ReasoningText(text) => text
                        .text()
                        .len()
                        .saturating_add(text.signature().map_or(0, str::len)),
                    ReasoningContentBlock::RedactedContent(bytes) => bytes.as_ref().len(),
                    _ => return Err(AgentError::Protocol(ProtocolFailure::UnsupportedContent)),
                },
                _ => return Err(AgentError::Protocol(ProtocolFailure::UnsupportedContent)),
            };
            bytes = bytes.saturating_add(count);
            if bytes > limits.transcript_bytes.get() {
                return Err(AgentError::Limit(Resource::TranscriptBytes));
            }
        }
    }
    Ok(())
}

fn document_bytes(document: &Document) -> usize {
    let mut pending = vec![document];
    let mut bytes = 0_usize;
    while let Some(value) = pending.pop() {
        bytes = bytes.saturating_add(1);
        match value {
            Document::Object(fields) => {
                for (key, value) in fields {
                    bytes = bytes.saturating_add(key.len());
                    pending.push(value);
                }
            }
            Document::Array(values) => pending.extend(values),
            Document::String(text) => bytes = bytes.saturating_add(text.len()),
            Document::Number(_) => bytes = bytes.saturating_add(8),
            Document::Bool(_) | Document::Null => {}
        }
    }
    bytes
}
