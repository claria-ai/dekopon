use std::{
    future::Future,
    num::{NonZeroU32, NonZeroUsize},
    ops::ControlFlow,
    time::Duration,
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

    fn record<'a>(
        &'a mut self,
        response: &'a BedrockResponse,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'a;

    fn model_failed(&mut self);

    fn execute<'a>(
        &'a mut self,
        call: &'a ToolUseBlock,
    ) -> impl Future<Output = Result<ToolResultBlock, Self::Error>> + Send + 'a;
}

pub trait Cancellation: Sync {
    fn is_stopped(&self) -> bool;
    fn stopped(&self) -> impl Future<Output = ()> + Send;
}

impl Cancellation for watch::Receiver<bool> {
    fn is_stopped(&self) -> bool {
        *self.borrow()
    }

    async fn stopped(&self) {
        let mut receiver = self.clone();
        loop {
            if *receiver.borrow_and_update() {
                return;
            }
            if receiver.changed().await.is_err() {
                std::future::pending::<()>().await;
            }
        }
    }
}

pub trait Session: Send {
    type Response: Send + Sync;
    type ModelError: Send;
    type Error: Send;
    type Outcome: Send;

    fn start_call(&mut self, index: u32) -> Result<(), Self::Error>;
    fn generate(&mut self)
    -> impl Future<Output = Result<Self::Response, Self::ModelError>> + Send;
    /// Persist completed-call accounting before the driver admits effects.
    /// This future is never dropped by driver cancellation.
    fn record(
        &mut self,
        response: &Self::Response,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
    /// Apply one accepted response. The host owns validation and commit boundaries.
    fn apply(
        &mut self,
        response: Self::Response,
    ) -> impl Future<Output = Result<ControlFlow<Self::Outcome>, Self::Error>> + Send;
    fn retry_delay(
        &self,
        error: &Self::ModelError,
        attempt: u32,
        elapsed: Duration,
    ) -> Option<Duration>;
    /// Preserve uncertainty from every physical failure, including recovered retries.
    fn model_failed(&mut self, error: &Self::ModelError);
    fn failed(&mut self, error: Self::ModelError, attempt: u32, elapsed: Duration) -> Self::Error;
    fn stopped(&mut self, in_flight: bool) -> Self::Error;
    fn call_limit(&mut self) -> Self::Error;
    /// A host may finish an already admitted batch or a durable finalization cut.
    fn cancellation_deferred(&self) -> bool {
        false
    }
}

#[derive(Clone, Copy)]
pub struct DriveLimits {
    pub model_calls: NonZeroU32,
    pub attempts_per_call: NonZeroU32,
}

pub async fn drive<S: Session, C: Cancellation>(
    session: &mut S,
    cancel: &C,
    limits: DriveLimits,
) -> Result<S::Outcome, S::Error> {
    for index in 1..=limits.model_calls.get() {
        if !session.cancellation_deferred() && cancel.is_stopped() {
            return Err(session.stopped(false));
        }
        session.start_call(index)?;
        let started = tokio::time::Instant::now();
        let mut attempt = 1;
        let response = loop {
            if !session.cancellation_deferred() && cancel.is_stopped() {
                return Err(session.stopped(false));
            }
            tracing::debug!(
                call = index,
                attempt,
                phase = "inference",
                "embedded session"
            );
            let generated = if session.cancellation_deferred() {
                session.generate().await
            } else {
                tokio::select! {
                    biased;
                    response = session.generate() => response,
                    () = cancel.stopped() => return Err(session.stopped(true)),
                }
            };
            match generated {
                Ok(response) => break response,
                Err(error) => {
                    session.model_failed(&error);
                    let elapsed = started.elapsed();
                    let delay = if attempt < limits.attempts_per_call.get() {
                        session.retry_delay(&error, attempt, elapsed)
                    } else {
                        None
                    };
                    let Some(delay) = delay else {
                        return Err(session.failed(error, attempt, elapsed));
                    };
                    if session.cancellation_deferred() {
                        tokio::time::sleep(delay).await;
                    } else {
                        tokio::select! {
                            biased;
                            () = cancel.stopped() => return Err(session.stopped(false)),
                            () = tokio::time::sleep(delay) => {},
                        }
                    }
                    attempt += 1;
                }
            }
        };
        // Completed usage must land even if Stop arrives while the receipt is being saved.
        tracing::debug!(call = index, attempt, phase = "receipt", "embedded session");
        session.record(&response).await?;
        if !session.cancellation_deferred() && cancel.is_stopped() {
            return Err(session.stopped(false));
        }
        // The host owns admitted effects and their commit boundary; never drop that future.
        tracing::debug!(call = index, attempt, phase = "effects", "embedded session");
        match session.apply(response).await? {
            ControlFlow::Break(outcome) => return Ok(outcome),
            ControlFlow::Continue(()) => {}
        }
    }
    if !session.cancellation_deferred() && cancel.is_stopped() {
        return Err(session.stopped(false));
    }
    Err(session.call_limit())
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
    request: BedrockRequest<'_>,
    limits: Limits,
    cancel: watch::Receiver<bool>,
    observe: &mut (dyn FnMut(&str) -> ControlFlow<()> + Send),
) -> Result<Outcome, AgentError<H::Error>> {
    let mut session = NativeSession {
        model,
        host,
        request,
        limits,
        cancel: &cancel,
        observe,
        model_calls: 0,
        tool_calls: 0,
    };
    drive(
        &mut session,
        &cancel,
        DriveLimits {
            model_calls: limits.model_calls,
            attempts_per_call: NonZeroU32::MIN,
        },
    )
    .await
}

struct NativeSession<'a, 'model, M, H> {
    model: &'a M,
    host: &'a mut H,
    request: BedrockRequest<'model>,
    limits: Limits,
    cancel: &'a watch::Receiver<bool>,
    observe: &'a mut (dyn FnMut(&str) -> ControlFlow<()> + Send),
    model_calls: u32,
    tool_calls: u32,
}

impl<M: InferenceModel, H: ToolHost> Session for NativeSession<'_, '_, M, H> {
    type Response = BedrockResponse;
    type ModelError = BedrockError;
    type Error = AgentError<H::Error>;
    type Outcome = Outcome;

    fn start_call(&mut self, index: u32) -> Result<(), Self::Error> {
        check_transcript(&self.request.messages, self.limits)?;
        self.model_calls = index;
        Ok(())
    }

    async fn generate(&mut self) -> Result<Self::Response, Self::ModelError> {
        self.model
            .generate(self.request.clone(), self.observe)
            .await
    }

    async fn record(&mut self, response: &Self::Response) -> Result<(), Self::Error> {
        self.host.record(response).await.map_err(AgentError::Host)
    }

    fn retry_delay(&self, _: &Self::ModelError, _: u32, _: Duration) -> Option<Duration> {
        None
    }

    fn model_failed(&mut self, _: &Self::ModelError) {
        self.host.model_failed();
    }

    fn failed(&mut self, error: Self::ModelError, _: u32, _: Duration) -> Self::Error {
        AgentError::Model(error)
    }

    fn stopped(&mut self, in_flight: bool) -> Self::Error {
        if in_flight {
            self.host.model_failed();
        }
        AgentError::Cancelled
    }

    fn call_limit(&mut self) -> Self::Error {
        AgentError::Limit(Resource::ModelCalls)
    }

    async fn apply(
        &mut self,
        response: Self::Response,
    ) -> Result<ControlFlow<Self::Outcome>, Self::Error> {
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
            model_calls = self.model_calls,
            tool_calls = call_count,
            latency_ms = response.latency_ms,
            "embedded model turn completed"
        );
        match (&response.stop_reason, calls.is_empty()) {
            (StopReason::EndTurn | StopReason::StopSequence, true) => {
                self.request.messages.push(response.message.clone());
                check_transcript(&self.request.messages, self.limits)?;
                if self.cancel.is_stopped() {
                    return Err(AgentError::Cancelled);
                }
                return Ok(ControlFlow::Break(Outcome {
                    transcript: std::mem::take(&mut self.request.messages),
                    response,
                    model_calls: self.model_calls,
                    tool_calls: self.tool_calls,
                }));
            }
            (StopReason::ToolUse, false) | (StopReason::MaxTokens, false) => {}
            (StopReason::ToolUse, true)
            | (StopReason::EndTurn | StopReason::StopSequence, false) => {
                return Err(AgentError::Protocol(ProtocolFailure::StopToolMismatch));
            }
            _ => return Err(AgentError::Stopped(response.stop_reason)),
        }
        if call_count > self.limits.tool_calls.get().saturating_sub(self.tool_calls) {
            return Err(AgentError::Limit(Resource::ToolCalls));
        }
        for (index, call) in calls.iter().enumerate() {
            if calls[..index]
                .iter()
                .any(|prior| prior.tool_use_id() == call.tool_use_id())
                || self
                    .request
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
            let offered = self.request.tools.as_ref().is_some_and(|config| {
                config.tools().iter().any(|tool| match tool {
                    Tool::ToolSpec(spec) => spec.name() == call.name(),
                    _ => false,
                })
            });
            if !offered {
                return Err(AgentError::Protocol(ProtocolFailure::ToolNotOffered));
            }
        }
        if self.request.messages.len().saturating_add(2) > self.limits.messages.get() {
            return Err(AgentError::Limit(Resource::Messages));
        }
        self.request.messages.push(response.message.clone());
        check_transcript(&self.request.messages, self.limits)?;
        let mut results = Vec::with_capacity(calls.len());
        for call in calls {
            if self.cancel.is_stopped() {
                return Err(AgentError::Cancelled);
            }
            // An admitted host effect owns its commit; cancellation only prevents the next effect.
            let result = self.host.execute(call).await.map_err(AgentError::Host)?;
            if result.tool_use_id() != call.tool_use_id() {
                return Err(AgentError::Protocol(ProtocolFailure::ToolResultId));
            }
            results.push(ContentBlock::ToolResult(result));
            self.tool_calls += 1;
        }
        let results = Message::builder()
            .role(ConversationRole::User)
            .set_content(Some(results))
            .build()
            .map_err(AgentError::Message)?;
        self.request.messages.push(results);
        Ok(ControlFlow::Continue(()))
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
