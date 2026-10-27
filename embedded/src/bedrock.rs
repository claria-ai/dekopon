use std::{
    collections::{BTreeMap, HashSet},
    ops::ControlFlow,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
    time::{Duration, Instant},
};

use aws_sdk_bedrockruntime::{
    error::{ProvideErrorMetadata, SdkError},
    operation::RequestId,
    types::{
        ContentBlock, ContentBlockDelta, ContentBlockStart, ConversationRole, ConverseStreamOutput,
        InferenceConfiguration, Message, ReasoningContentBlock, ReasoningContentBlockDelta,
        ReasoningTextBlock, StopReason, SystemContentBlock, TokenUsage, ToolConfiguration,
        ToolUseBlock,
    },
};
use aws_smithy_types::{Blob, Document, Number};
use thiserror::Error;

#[derive(Clone, Copy)]
pub struct StreamLimits {
    first_frame: Duration,
    idle: Duration,
    max_response_bytes: usize,
    max_blocks: usize,
    max_tool_calls: usize,
}

impl StreamLimits {
    pub fn new(
        first_frame: Duration,
        idle: Duration,
        max_response_bytes: usize,
        max_blocks: usize,
        max_tool_calls: usize,
    ) -> Result<Self, BedrockError> {
        if first_frame.is_zero()
            || idle.is_zero()
            || max_response_bytes == 0
            || max_blocks == 0
            || max_tool_calls == 0
        {
            return Err(BedrockError::InvalidLimits);
        }
        Ok(Self {
            first_frame,
            idle,
            max_response_bytes,
            max_blocks,
            max_tool_calls,
        })
    }
}

impl Default for StreamLimits {
    fn default() -> Self {
        Self {
            first_frame: Duration::from_secs(600),
            idle: Duration::from_secs(600),
            max_response_bytes: 8 * 1024 * 1024,
            max_blocks: 1024,
            max_tool_calls: 100,
        }
    }
}

#[derive(Clone)]
pub struct BedrockRequest<'a> {
    pub model_id: &'a str,
    pub system: Vec<SystemContentBlock>,
    pub messages: Vec<Message>,
    pub tools: Option<ToolConfiguration>,
    pub inference: InferenceConfiguration,
    pub additional_fields: Option<Document>,
    pub limits: StreamLimits,
}

pub struct BedrockResponse {
    pub message: Message,
    pub stop_reason: StopReason,
    pub usage: Option<TokenUsage>,
    pub latency_ms: u64,
    /// SDK signing-stage attempts, not a provider billing count. More than one
    /// means the final response's usage cannot account for earlier attempts.
    pub sdk_attempts: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailurePhase {
    AwaitingResponse,
    Streaming,
    AfterTerminal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterruptionKind {
    Timeout,
    Transport,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum ProtocolFailure {
    #[error("response ended without messageStop")]
    MissingTerminal,
    #[error("response repeated messageStop")]
    DuplicateTerminal,
    #[error("response content arrived after messageStop")]
    ContentAfterTerminal,
    #[error("response block index is invalid")]
    InvalidIndex,
    #[error("response block changed its content type")]
    ConflictingBlock,
    #[error("response delta is missing its content")]
    MissingDelta,
    #[error("response contains a content type unsupported by this text-and-tool backend")]
    UnsupportedContent,
    #[error("response message start is invalid or repeated")]
    InvalidStart,
    #[error("response tool delta arrived without its start")]
    ToolWithoutStart,
    #[error("response repeated a tool start")]
    DuplicateToolStart,
    #[error("response tool identity is empty or duplicated")]
    InvalidToolIdentity,
    #[error("response tool input is not valid JSON")]
    InvalidToolInput,
    #[error("response reasoning mixes plaintext and redacted content")]
    MixedReasoning,
    #[error("response exceeded its content-byte bound")]
    TooManyBytes,
    #[error("response exceeded its block bound")]
    TooManyBlocks,
    #[error("response exceeded its tool-call bound")]
    TooManyTools,
    #[error("response has no visible text or tool call")]
    EmptyResponse,
    #[error("response could not be represented as an SDK message")]
    InvalidMessage,
}

#[derive(Debug, Error)]
pub enum BedrockError {
    #[error("stream limits must be greater than zero")]
    InvalidLimits,
    #[error("model id and a positive output token limit are required")]
    InvalidRequest,
    #[error("model request was cancelled")]
    Cancelled,
    #[error("Bedrock interrupted in {phase:?}: {kind:?}")]
    Interrupted {
        phase: FailurePhase,
        kind: InterruptionKind,
    },
    #[error(
        "Bedrock service failure in {phase:?}: {code}, status {status:?}, request {request_id:?}"
    )]
    Service {
        phase: FailurePhase,
        code: String,
        status: Option<u16>,
        request_id: Option<String>,
    },
    #[error(transparent)]
    Protocol(#[from] ProtocolFailure),
}

pub struct BedrockClient {
    client: aws_sdk_bedrockruntime::Client,
}

impl BedrockClient {
    pub fn new(config: &aws_config::SdkConfig) -> Self {
        Self {
            client: aws_sdk_bedrockruntime::Client::new(config),
        }
    }

    pub fn from_client(client: aws_sdk_bedrockruntime::Client) -> Self {
        Self { client }
    }

    pub async fn converse(
        &self,
        request: BedrockRequest<'_>,
        observe: &mut (dyn FnMut(&str) -> ControlFlow<()> + Send),
    ) -> Result<BedrockResponse, BedrockError> {
        if request.model_id.trim().is_empty() || request.inference.max_tokens().unwrap_or(0) <= 0 {
            return Err(BedrockError::InvalidRequest);
        }
        let started = Instant::now();
        let attempts = Arc::new(AtomicU32::new(0));
        let observed_attempts = Arc::clone(&attempts);
        let send = self
            .client
            .converse_stream()
            .model_id(request.model_id)
            .set_system(Some(request.system))
            .set_messages(Some(request.messages))
            .set_tool_config(request.tools)
            .inference_config(request.inference)
            .set_additional_model_request_fields(request.additional_fields)
            .customize()
            .mutate_request(move |_| {
                observed_attempts.fetch_add(1, Ordering::Relaxed);
            })
            .send();
        let mut response = tokio::time::timeout(request.limits.first_frame, send)
            .await
            .map_err(|_| BedrockError::Interrupted {
                phase: FailurePhase::AwaitingResponse,
                kind: InterruptionKind::Timeout,
            })?
            .map_err(|error| {
                service_error(FailurePhase::AwaitingResponse, error, |response| {
                    Some(response.status().as_u16())
                })
            })?;
        let mut collector = StreamCollector::new(request.limits);
        loop {
            let phase = collector.phase();
            let event = tokio::time::timeout(request.limits.idle, response.stream.recv())
                .await
                .map_err(|_| BedrockError::Interrupted {
                    phase,
                    kind: InterruptionKind::Timeout,
                })?
                .map_err(|error| stream_service_error(phase, error))?;
            match event {
                Some(event) => collector.absorb(event, observe)?,
                None => break,
            }
        }
        let mut result = collector.finish()?;
        result.latency_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        result.sdk_attempts = attempts.load(Ordering::Relaxed);
        tracing::info!(
            sdk_attempts = result.sdk_attempts,
            latency_ms = result.latency_ms,
            "embedded model exchange completed"
        );
        Ok(result)
    }
}

fn stream_service_error<R>(
    phase: FailurePhase,
    error: SdkError<aws_sdk_bedrockruntime::types::error::ConverseStreamOutputError, R>,
) -> BedrockError {
    use aws_sdk_bedrockruntime::types::error::ConverseStreamOutputError;

    if let Some(service) = error.as_service_error() {
        let code = match service {
            ConverseStreamOutputError::ModelStreamErrorException(_) => {
                Some("ModelStreamErrorException")
            }
            ConverseStreamOutputError::InternalServerException(_) => {
                Some("InternalServerException")
            }
            ConverseStreamOutputError::ThrottlingException(_) => Some("ThrottlingException"),
            ConverseStreamOutputError::ServiceUnavailableException(_) => {
                Some("ServiceUnavailableException")
            }
            ConverseStreamOutputError::ValidationException(_) => Some("ValidationException"),
            _ => None,
        };
        if let Some(code) = code {
            return BedrockError::Service {
                phase,
                code: code.to_owned(),
                status: None,
                request_id: service.meta().request_id().map(str::to_owned),
            };
        }
    }
    service_error(phase, error, |_| None)
}

fn service_error<E, R>(
    phase: FailurePhase,
    error: SdkError<E, R>,
    status: impl FnOnce(&R) -> Option<u16>,
) -> BedrockError
where
    E: ProvideErrorMetadata,
{
    if let Some(service) = error.as_service_error() {
        return BedrockError::Service {
            phase,
            code: service.code().unwrap_or("UnknownServiceError").to_owned(),
            status: error.raw_response().and_then(status),
            request_id: service.meta().request_id().map(str::to_owned),
        };
    }
    match error {
        SdkError::ConstructionFailure(_) => BedrockError::InvalidRequest,
        SdkError::TimeoutError(_) => BedrockError::Interrupted {
            phase,
            kind: InterruptionKind::Timeout,
        },
        _ => BedrockError::Interrupted {
            phase,
            kind: InterruptionKind::Transport,
        },
    }
}

enum PartialBlock {
    Text(String),
    Tool {
        id: String,
        name: String,
        input: String,
    },
    Reasoning {
        text: String,
        signature: String,
        redacted: Vec<u8>,
    },
}

pub struct StreamCollector {
    limits: StreamLimits,
    bytes: usize,
    blocks: BTreeMap<i32, PartialBlock>,
    tool_ids: HashSet<String>,
    started: bool,
    stop: Option<StopReason>,
    usage: Option<TokenUsage>,
}

impl StreamCollector {
    pub fn new(limits: StreamLimits) -> Self {
        Self {
            limits,
            bytes: 0,
            blocks: BTreeMap::new(),
            tool_ids: HashSet::new(),
            started: false,
            stop: None,
            usage: None,
        }
    }

    pub fn phase(&self) -> FailurePhase {
        if self.stop.is_some() {
            FailurePhase::AfterTerminal
        } else {
            FailurePhase::Streaming
        }
    }

    pub fn absorb(
        &mut self,
        event: ConverseStreamOutput,
        observe: &mut (dyn FnMut(&str) -> ControlFlow<()> + Send),
    ) -> Result<(), BedrockError> {
        match event {
            ConverseStreamOutput::ContentBlockStart(event) => {
                self.content_index(event.content_block_index())?;
                if matches!(
                    event.start(),
                    Some(ContentBlockStart::Image(_) | ContentBlockStart::ToolResult(_))
                ) {
                    return Err(ProtocolFailure::UnsupportedContent.into());
                }
                if let Some(ContentBlockStart::ToolUse(tool)) = event.start() {
                    let index = event.content_block_index();
                    if self.blocks.contains_key(&index) {
                        return Err(ProtocolFailure::DuplicateToolStart.into());
                    }
                    if tool.tool_use_id().trim().is_empty()
                        || tool.name().trim().is_empty()
                        || self.tool_ids.contains(tool.tool_use_id())
                    {
                        return Err(ProtocolFailure::InvalidToolIdentity.into());
                    }
                    if self.tool_ids.len() >= self.limits.max_tool_calls {
                        return Err(ProtocolFailure::TooManyTools.into());
                    }
                    self.account(tool.tool_use_id().len().saturating_add(tool.name().len()))?;
                    self.tool_ids.insert(tool.tool_use_id().to_owned());
                    self.blocks.insert(
                        index,
                        PartialBlock::Tool {
                            id: tool.tool_use_id().to_owned(),
                            name: tool.name().to_owned(),
                            input: String::new(),
                        },
                    );
                }
            }
            ConverseStreamOutput::ContentBlockDelta(event) => {
                let index = event.content_block_index();
                self.content_index(index)?;
                match event.delta() {
                    Some(ContentBlockDelta::Text(text)) => {
                        self.account(text.len())?;
                        let block = self
                            .blocks
                            .entry(index)
                            .or_insert_with(|| PartialBlock::Text(String::new()));
                        match block {
                            PartialBlock::Text(value) => value.push_str(text),
                            PartialBlock::Tool { .. } | PartialBlock::Reasoning { .. } => {
                                return Err(ProtocolFailure::ConflictingBlock.into());
                            }
                        }
                        if observe(text).is_break() {
                            return Err(BedrockError::Cancelled);
                        }
                    }
                    Some(ContentBlockDelta::ToolUse(delta)) => {
                        self.account(delta.input().len())?;
                        match self.blocks.get_mut(&index) {
                            Some(PartialBlock::Tool { input, .. }) => input.push_str(delta.input()),
                            Some(PartialBlock::Text(_) | PartialBlock::Reasoning { .. }) => {
                                return Err(ProtocolFailure::ConflictingBlock.into());
                            }
                            None => return Err(ProtocolFailure::ToolWithoutStart.into()),
                        }
                    }
                    Some(ContentBlockDelta::ReasoningContent(delta)) => {
                        self.reasoning(index, delta)?
                    }
                    Some(
                        ContentBlockDelta::Citation(_)
                        | ContentBlockDelta::Image(_)
                        | ContentBlockDelta::ToolResult(_),
                    ) => {
                        return Err(ProtocolFailure::UnsupportedContent.into());
                    }
                    None => return Err(ProtocolFailure::MissingDelta.into()),
                    Some(_) => tracing::debug!("ignored unknown Bedrock content delta"),
                }
            }
            ConverseStreamOutput::MessageStop(event) => {
                if self.stop.is_some() {
                    return Err(ProtocolFailure::DuplicateTerminal.into());
                }
                self.stop = Some(event.stop_reason().clone());
            }
            ConverseStreamOutput::Metadata(event) => self.usage = event.usage().cloned(),
            ConverseStreamOutput::MessageStart(event) => {
                if self.started
                    || self.stop.is_some()
                    || !self.blocks.is_empty()
                    || event.role() != &ConversationRole::Assistant
                {
                    return Err(ProtocolFailure::InvalidStart.into());
                }
                self.started = true;
            }
            ConverseStreamOutput::ContentBlockStop(_) => {}
            _ => tracing::debug!("ignored unknown Bedrock stream event"),
        }
        Ok(())
    }

    fn content_index(&self, index: i32) -> Result<(), ProtocolFailure> {
        if self.stop.is_some() {
            return Err(ProtocolFailure::ContentAfterTerminal);
        }
        if index < 0 {
            return Err(ProtocolFailure::InvalidIndex);
        }
        if !self.blocks.contains_key(&index) && self.blocks.len() >= self.limits.max_blocks {
            return Err(ProtocolFailure::TooManyBlocks);
        }
        Ok(())
    }

    fn account(&mut self, bytes: usize) -> Result<(), ProtocolFailure> {
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or(ProtocolFailure::TooManyBytes)?;
        if self.bytes > self.limits.max_response_bytes {
            return Err(ProtocolFailure::TooManyBytes);
        }
        Ok(())
    }

    fn reasoning(
        &mut self,
        index: i32,
        delta: &ReasoningContentBlockDelta,
    ) -> Result<(), ProtocolFailure> {
        let bytes = match delta {
            ReasoningContentBlockDelta::Text(text)
            | ReasoningContentBlockDelta::Signature(text) => text.len(),
            ReasoningContentBlockDelta::RedactedContent(bytes) => bytes.as_ref().len(),
            _ => {
                tracing::debug!("ignored unknown Bedrock reasoning delta");
                return Ok(());
            }
        };
        self.account(bytes)?;
        let block = self
            .blocks
            .entry(index)
            .or_insert_with(|| PartialBlock::Reasoning {
                text: String::new(),
                signature: String::new(),
                redacted: Vec::new(),
            });
        match block {
            PartialBlock::Reasoning {
                text,
                signature,
                redacted,
            } => {
                match delta {
                    ReasoningContentBlockDelta::Text(part) => text.push_str(part),
                    ReasoningContentBlockDelta::Signature(part) => signature.push_str(part),
                    ReasoningContentBlockDelta::RedactedContent(part) => {
                        redacted.extend_from_slice(part.as_ref())
                    }
                    _ => {}
                }
                if !redacted.is_empty() && (!text.is_empty() || !signature.is_empty()) {
                    return Err(ProtocolFailure::MixedReasoning);
                }
                Ok(())
            }
            PartialBlock::Text(_) | PartialBlock::Tool { .. } => {
                Err(ProtocolFailure::ConflictingBlock)
            }
        }
    }

    pub fn finish(self) -> Result<BedrockResponse, BedrockError> {
        let stop_reason = self.stop.ok_or(ProtocolFailure::MissingTerminal)?;
        let mut content = Vec::with_capacity(self.blocks.len());
        for block in self.blocks.into_values() {
            match block {
                PartialBlock::Text(text) => {
                    if !text.is_empty() {
                        content.push(ContentBlock::Text(text));
                    }
                }
                PartialBlock::Tool { id, name, input } => {
                    let input = if input.trim().is_empty() {
                        "{}"
                    } else {
                        input.as_str()
                    };
                    let value: serde_json::Value = serde_json::from_str(input)
                        .map_err(|_| ProtocolFailure::InvalidToolInput)?;
                    content.push(ContentBlock::ToolUse(
                        ToolUseBlock::builder()
                            .tool_use_id(id)
                            .name(name)
                            .input(document(value)?)
                            .build()
                            .map_err(|_| ProtocolFailure::InvalidMessage)?,
                    ));
                }
                PartialBlock::Reasoning {
                    text,
                    signature,
                    redacted,
                } => {
                    if !redacted.is_empty() {
                        content.push(ContentBlock::ReasoningContent(
                            ReasoningContentBlock::RedactedContent(Blob::new(redacted)),
                        ));
                    } else if !text.is_empty() || !signature.is_empty() {
                        let signature = (!signature.is_empty()).then_some(signature);
                        content.push(ContentBlock::ReasoningContent(
                            ReasoningContentBlock::ReasoningText(
                                ReasoningTextBlock::builder()
                                    .text(text)
                                    .set_signature(signature)
                                    .build()
                                    .map_err(|_| ProtocolFailure::InvalidMessage)?,
                            ),
                        ));
                    }
                }
            }
        }
        if !content
            .iter()
            .any(|block| matches!(block, ContentBlock::Text(_) | ContentBlock::ToolUse(_)))
        {
            return Err(ProtocolFailure::EmptyResponse.into());
        }
        let message = Message::builder()
            .role(ConversationRole::Assistant)
            .set_content(Some(content))
            .build()
            .map_err(|_| ProtocolFailure::InvalidMessage)?;
        Ok(BedrockResponse {
            message,
            stop_reason,
            usage: self.usage,
            latency_ms: 0,
            sdk_attempts: 1,
        })
    }
}

fn document(value: serde_json::Value) -> Result<Document, ProtocolFailure> {
    Ok(match value {
        serde_json::Value::Null => Document::Null,
        serde_json::Value::Bool(value) => Document::Bool(value),
        serde_json::Value::String(value) => Document::String(value),
        serde_json::Value::Array(values) => {
            Document::Array(values.into_iter().map(document).collect::<Result<_, _>>()?)
        }
        serde_json::Value::Object(values) => Document::Object(
            values
                .into_iter()
                .map(|(key, value)| document(value).map(|value| (key, value)))
                .collect::<Result<_, _>>()?,
        ),
        serde_json::Value::Number(value) => {
            Document::Number(match (value.as_u64(), value.as_i64()) {
                (Some(value), _) => Number::PosInt(value),
                (None, Some(value)) => Number::NegInt(value),
                (None, None) => {
                    Number::Float(value.as_f64().ok_or(ProtocolFailure::InvalidToolInput)?)
                }
            })
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use aws_sdk_bedrockruntime::types::error::{
        ConverseStreamOutputError, InternalServerException, ModelStreamErrorException,
        ServiceUnavailableException, ThrottlingException, ValidationException,
    };
    use aws_smithy_types::error::ErrorMetadata;

    #[test]
    fn typed_stream_exceptions_retain_their_code_when_metadata_omits_it() {
        for phase in [FailurePhase::Streaming, FailurePhase::AfterTerminal] {
            let meta = ErrorMetadata::builder()
                .custom("aws_request_id", "synthetic-request")
                .build();
            let cases = [
                (
                    ConverseStreamOutputError::ModelStreamErrorException(
                        ModelStreamErrorException::builder()
                            .meta(meta.clone())
                            .build(),
                    ),
                    "ModelStreamErrorException",
                ),
                (
                    ConverseStreamOutputError::InternalServerException(
                        InternalServerException::builder()
                            .meta(meta.clone())
                            .build(),
                    ),
                    "InternalServerException",
                ),
                (
                    ConverseStreamOutputError::ThrottlingException(
                        ThrottlingException::builder().meta(meta.clone()).build(),
                    ),
                    "ThrottlingException",
                ),
                (
                    ConverseStreamOutputError::ServiceUnavailableException(
                        ServiceUnavailableException::builder()
                            .meta(meta.clone())
                            .build(),
                    ),
                    "ServiceUnavailableException",
                ),
                (
                    ConverseStreamOutputError::ValidationException(
                        ValidationException::builder().meta(meta).build(),
                    ),
                    "ValidationException",
                ),
            ];
            for (service, expected) in cases {
                assert!(service.code().is_none());
                let error = stream_service_error(phase, SdkError::service_error(service, ()));
                let BedrockError::Service {
                    phase: actual,
                    code,
                    status,
                    request_id,
                } = error
                else {
                    panic!("typed service failure retained");
                };
                assert_eq!(actual, phase);
                assert_eq!(code, expected);
                assert_eq!(status, None);
                assert_eq!(request_id.as_deref(), Some("synthetic-request"));
            }
        }
    }
}
