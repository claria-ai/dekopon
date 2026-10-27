use std::{ops::ControlFlow, time::Duration};

use aws_smithy_types::{Blob, Document, Number};
use dekopon_embedded::{
    bedrock::{BedrockError, FailurePhase, ProtocolFailure, StreamCollector, StreamLimits},
    sdk::types::{
        ContentBlock, ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockStart,
        ContentBlockStartEvent, ConverseStreamMetadataEvent, ConverseStreamOutput,
        MessageStopEvent, ReasoningContentBlock, ReasoningContentBlockDelta, StopReason,
        TokenUsage, ToolUseBlockDelta, ToolUseBlockStart,
    },
};

fn collector() -> StreamCollector {
    StreamCollector::new(StreamLimits::default())
}

fn emit(collector: &mut StreamCollector, event: ConverseStreamOutput) -> Result<(), BedrockError> {
    collector.absorb(event, &mut |_| ControlFlow::Continue(()))
}

fn delta(index: i32, content: ContentBlockDelta) -> ConverseStreamOutput {
    ConverseStreamOutput::ContentBlockDelta(
        ContentBlockDeltaEvent::builder()
            .content_block_index(index)
            .delta(content)
            .build()
            .expect("fixture delta"),
    )
}

fn text(index: i32, value: &str) -> ConverseStreamOutput {
    delta(index, ContentBlockDelta::Text(value.into()))
}

fn tool(index: i32, id: &str) -> ConverseStreamOutput {
    ConverseStreamOutput::ContentBlockStart(
        ContentBlockStartEvent::builder()
            .content_block_index(index)
            .start(ContentBlockStart::ToolUse(
                ToolUseBlockStart::builder()
                    .tool_use_id(id)
                    .name("read_evidence")
                    .build()
                    .expect("fixture tool"),
            ))
            .build()
            .expect("fixture start"),
    )
}

fn input(index: i32, value: &str) -> ConverseStreamOutput {
    delta(
        index,
        ContentBlockDelta::ToolUse(
            ToolUseBlockDelta::builder()
                .input(value)
                .build()
                .expect("fixture tool input"),
        ),
    )
}

fn stop(reason: StopReason) -> ConverseStreamOutput {
    ConverseStreamOutput::MessageStop(
        MessageStopEvent::builder()
            .stop_reason(reason)
            .build()
            .expect("fixture stop"),
    )
}

#[test]
fn tool_json_and_signed_reasoning_retain_exact_content_in_block_order() {
    let mut collector = collector();
    let mut observed = String::new();
    collector
        .absorb(text(2, "Visible résumé"), &mut |part| {
            observed.push_str(part);
            ControlFlow::Continue(())
        })
        .expect("text");
    emit(
        &mut collector,
        delta(
            0,
            ContentBlockDelta::ReasoningContent(ReasoningContentBlockDelta::Text(
                "private thought".into(),
            )),
        ),
    )
    .expect("reason");
    for part in ["signed-", "continuation"] {
        emit(
            &mut collector,
            delta(
                0,
                ContentBlockDelta::ReasoningContent(ReasoningContentBlockDelta::Signature(
                    part.into(),
                )),
            ),
        )
        .expect("signature");
    }
    emit(&mut collector, tool(1, "call-1")).expect("start");
    for part in [
        "{\"file\":\"ré",
        "sumé\",\"values\":[",
        "-2,3,0.25,true,null]}",
    ] {
        emit(&mut collector, input(1, part)).expect("input");
    }
    emit(&mut collector, stop(StopReason::ToolUse)).expect("stop");
    let output = collector.finish().expect("complete");
    assert_eq!(observed, "Visible résumé");
    let blocks = output.message.content();
    let ContentBlock::ReasoningContent(ReasoningContentBlock::ReasoningText(reason)) = &blocks[0]
    else {
        panic!("reasoning retained");
    };
    assert_eq!(reason.text(), "private thought");
    assert_eq!(reason.signature(), Some("signed-continuation"));
    let ContentBlock::ToolUse(call) = &blocks[1] else {
        panic!("tool retained");
    };
    assert_eq!(call.tool_use_id(), "call-1");
    let Document::Object(input) = call.input() else {
        panic!("object retained");
    };
    assert_eq!(input["file"], Document::String("résumé".into()));
    assert_eq!(
        input["values"],
        Document::Array(vec![
            Document::Number(Number::NegInt(-2)),
            Document::Number(Number::PosInt(3)),
            Document::Number(Number::Float(0.25)),
            Document::Bool(true),
            Document::Null
        ])
    );
    assert_eq!(blocks[2], ContentBlock::Text("Visible résumé".into()));
    assert_eq!(output.stop_reason, StopReason::ToolUse);
    assert!(output.usage.is_none());
}

#[test]
fn redacted_reasoning_is_concatenated_without_becoming_visible() {
    let mut collector = collector();
    for part in [b"first".as_slice(), b"second".as_slice()] {
        emit(
            &mut collector,
            delta(
                0,
                ContentBlockDelta::ReasoningContent(ReasoningContentBlockDelta::RedactedContent(
                    Blob::new(part),
                )),
            ),
        )
        .expect("reasoning");
    }
    emit(&mut collector, text(1, "Answer")).expect("text");
    emit(&mut collector, stop(StopReason::EndTurn)).expect("stop");
    let output = collector.finish().expect("complete");
    let ContentBlock::ReasoningContent(ReasoningContentBlock::RedactedContent(bytes)) =
        &output.message.content()[0]
    else {
        panic!("redacted content retained");
    };
    assert_eq!(bytes.as_ref(), b"firstsecond");
}

#[test]
fn terminal_usage_keeps_bedrock_cache_buckets_without_normalization() {
    let mut collector = collector();
    emit(&mut collector, text(0, "Answer")).expect("text");
    assert_eq!(collector.phase(), FailurePhase::Streaming);
    emit(&mut collector, stop(StopReason::EndTurn)).expect("stop");
    assert_eq!(collector.phase(), FailurePhase::AfterTerminal);
    let usage = TokenUsage::builder()
        .input_tokens(11)
        .output_tokens(7)
        .total_tokens(18)
        .cache_read_input_tokens(100)
        .cache_write_input_tokens(50)
        .build()
        .expect("usage");
    emit(
        &mut collector,
        ConverseStreamOutput::Metadata(
            ConverseStreamMetadataEvent::builder()
                .usage(usage.clone())
                .build(),
        ),
    )
    .expect("metadata");
    assert_eq!(collector.finish().expect("complete").usage, Some(usage));
}

#[test]
fn completed_tools_survive_max_tokens_but_malformed_input_never_executes() {
    for reason in [StopReason::MaxTokens, StopReason::EndTurn] {
        let mut collector = collector();
        emit(&mut collector, tool(0, "call-1")).expect("start");
        emit(&mut collector, input(0, "{\"offset\":0}")).expect("input");
        emit(&mut collector, stop(reason.clone())).expect("stop");
        let output = collector.finish().expect("complete tool");
        assert_eq!(output.stop_reason, reason);
        assert!(matches!(
            output.message.content()[0],
            ContentBlock::ToolUse(_)
        ));
    }
    let mut collector = collector();
    emit(&mut collector, tool(0, "call-2")).expect("start");
    emit(&mut collector, input(0, "{\"PRIVATE_FIXTURE_SENTINEL\":")).expect("input");
    emit(&mut collector, stop(StopReason::MaxTokens)).expect("stop");
    let error = collector.finish().err().expect("malformed input rejected");
    assert!(matches!(
        error,
        BedrockError::Protocol(ProtocolFailure::InvalidToolInput)
    ));
    assert!(!format!("{error:?} {error}").contains("PRIVATE_FIXTURE_SENTINEL"));
}

#[test]
fn absent_terminal_is_never_a_successful_partial_response() {
    let mut collector = collector();
    emit(&mut collector, text(0, "partial")).expect("text");
    assert!(matches!(
        collector.finish(),
        Err(BedrockError::Protocol(ProtocolFailure::MissingTerminal))
    ));
}

#[test]
fn response_bounds_count_utf8_bytes_empty_blocks_and_tool_headers() {
    let limits =
        StreamLimits::new(Duration::from_secs(1), Duration::from_secs(1), 3, 1, 1).expect("limits");
    let mut bytes = StreamCollector::new(limits);
    assert!(matches!(
        emit(&mut bytes, text(0, "éé")),
        Err(BedrockError::Protocol(ProtocolFailure::TooManyBytes))
    ));
    let mut blocks = StreamCollector::new(limits);
    emit(&mut blocks, text(0, "")).expect("empty first block");
    assert!(matches!(
        emit(&mut blocks, text(1, "")),
        Err(BedrockError::Protocol(ProtocolFailure::TooManyBlocks))
    ));
    let mut headers = StreamCollector::new(limits);
    assert!(matches!(
        emit(&mut headers, tool(0, "call-1")),
        Err(BedrockError::Protocol(ProtocolFailure::TooManyBytes))
    ));
}

#[test]
fn tools_require_a_unique_start_and_obey_the_call_limit() {
    let mut missing = collector();
    assert!(matches!(
        emit(&mut missing, input(0, "{}")),
        Err(BedrockError::Protocol(ProtocolFailure::ToolWithoutStart))
    ));
    let mut repeated = collector();
    emit(&mut repeated, tool(0, "call-1")).expect("start");
    assert!(matches!(
        emit(&mut repeated, tool(1, "call-1")),
        Err(BedrockError::Protocol(ProtocolFailure::InvalidToolIdentity))
    ));
    let limits = StreamLimits::new(Duration::from_secs(1), Duration::from_secs(1), 1024, 10, 1)
        .expect("limits");
    let mut limited = StreamCollector::new(limits);
    emit(&mut limited, tool(0, "call-1")).expect("start");
    assert!(matches!(
        emit(&mut limited, tool(1, "call-2")),
        Err(BedrockError::Protocol(ProtocolFailure::TooManyTools))
    ));
}

#[test]
fn an_observer_break_cancels_the_response() {
    let mut collector = collector();
    assert!(matches!(
        collector.absorb(text(0, "partial"), &mut |_| ControlFlow::Break(())),
        Err(BedrockError::Cancelled)
    ));
}

#[test]
fn a_terminal_response_rejects_later_content_or_another_terminal() {
    let mut collector = collector();
    emit(&mut collector, stop(StopReason::EndTurn)).expect("stop");
    assert!(matches!(
        emit(&mut collector, text(0, "late")),
        Err(BedrockError::Protocol(
            ProtocolFailure::ContentAfterTerminal
        ))
    ));
    assert!(matches!(
        emit(&mut collector, stop(StopReason::EndTurn)),
        Err(BedrockError::Protocol(ProtocolFailure::DuplicateTerminal))
    ));
}

#[test]
fn zero_stream_limits_are_rejected() {
    assert!(matches!(
        StreamLimits::new(Duration::ZERO, Duration::from_secs(1), 1, 1, 1),
        Err(BedrockError::InvalidLimits)
    ));
}

#[test]
fn known_malformed_events_are_rejected_with_typed_failures() {
    let mut collector = collector();
    let empty = ConverseStreamOutput::ContentBlockDelta(
        ContentBlockDeltaEvent::builder()
            .content_block_index(0)
            .build()
            .expect("fixture empty delta"),
    );
    assert!(matches!(
        emit(&mut collector, empty),
        Err(BedrockError::Protocol(ProtocolFailure::MissingDelta))
    ));
    let start = ConverseStreamOutput::MessageStart(
        dekopon_embedded::sdk::types::MessageStartEvent::builder()
            .role(dekopon_embedded::sdk::types::ConversationRole::User)
            .build()
            .expect("fixture user start"),
    );
    assert!(matches!(
        emit(&mut collector, start),
        Err(BedrockError::Protocol(ProtocolFailure::InvalidStart))
    ));
}

#[test]
fn tool_schema_errors_remain_available_for_host_repair() {
    let mut collector = collector();
    emit(&mut collector, tool(0, "call-1")).expect("start");
    emit(&mut collector, input(0, "[\"wrong shape\"]")).expect("input");
    emit(&mut collector, stop(StopReason::ToolUse)).expect("stop");
    let output = collector
        .finish()
        .expect("valid protocol despite schema error");
    let ContentBlock::ToolUse(tool) = &output.message.content()[0] else {
        panic!("tool retained");
    };
    assert_eq!(
        tool.input(),
        &Document::Array(vec![Document::String("wrong shape".into())])
    );
}

#[test]
fn known_unsupported_content_is_refused_instead_of_silently_lost() {
    use dekopon_embedded::sdk::types::{CitationsDelta, ImageBlockDelta, ToolResultBlockStart};
    for content in [
        ContentBlockDelta::Citation(CitationsDelta::builder().build()),
        ContentBlockDelta::Image(ImageBlockDelta::builder().build()),
        ContentBlockDelta::ToolResult(Vec::new()),
    ] {
        assert!(matches!(
            emit(&mut collector(), delta(0, content)),
            Err(BedrockError::Protocol(ProtocolFailure::UnsupportedContent))
        ));
    }
    let start = ConverseStreamOutput::ContentBlockStart(
        ContentBlockStartEvent::builder()
            .content_block_index(0)
            .start(ContentBlockStart::ToolResult(
                ToolResultBlockStart::builder()
                    .tool_use_id("call-1")
                    .build()
                    .expect("fixture native result"),
            ))
            .build()
            .expect("fixture start"),
    );
    assert!(matches!(
        emit(&mut collector(), start),
        Err(BedrockError::Protocol(ProtocolFailure::UnsupportedContent))
    ));
}
