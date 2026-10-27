#![allow(
    clippy::disallowed_types,
    reason = "tests own and synchronously inspect scripted calls"
)]

use std::{
    collections::{HashMap, VecDeque},
    convert::Infallible,
    num::{NonZeroU32, NonZeroUsize},
    ops::ControlFlow,
    sync::Mutex,
};

use aws_smithy_types::Document;
use dekopon_embedded::{
    agent::{self, AgentError, InferenceModel, Limits, ProtocolFailure, Resource, ToolHost},
    bedrock::{BedrockError, BedrockRequest, BedrockResponse, StreamLimits},
    sdk::types::{
        ContentBlock, ConversationRole, InferenceConfiguration, Message, ReasoningContentBlock,
        ReasoningTextBlock, StopReason, Tool, ToolConfiguration, ToolInputSchema, ToolResultBlock,
        ToolResultContentBlock, ToolResultStatus, ToolSpecification, ToolUseBlock,
    },
};
use tokio::sync::watch;

struct ScriptedModel {
    turns: Mutex<VecDeque<BedrockResponse>>,
    requests: Mutex<Vec<Vec<Message>>>,
}

impl ScriptedModel {
    fn new(turns: Vec<BedrockResponse>) -> Self {
        Self {
            turns: Mutex::new(turns.into()),
            requests: Mutex::new(Vec::new()),
        }
    }
}

impl InferenceModel for ScriptedModel {
    async fn generate<'a>(
        &'a self,
        request: BedrockRequest<'a>,
        observe: &'a mut (dyn FnMut(&str) -> ControlFlow<()> + Send),
    ) -> Result<BedrockResponse, BedrockError> {
        self.requests
            .lock()
            .expect("requests")
            .push(request.messages);
        let response = self
            .turns
            .lock()
            .expect("script")
            .pop_front()
            .expect("scripted turn");
        for block in response.message.content() {
            if let ContentBlock::Text(text) = block {
                assert!(observe(text).is_continue());
            }
        }
        Ok(response)
    }
}

#[derive(Default)]
struct DocumentHost {
    revisions: Vec<String>,
    reads: usize,
    cancel_after_commit: Option<watch::Sender<bool>>,
}

impl ToolHost for DocumentHost {
    type Error = Infallible;

    async fn execute(&mut self, call: &ToolUseBlock) -> Result<ToolResultBlock, Self::Error> {
        let (text, status) = match call.name() {
            "read_record" => {
                self.reads += 1;
                if self.reads == 1 {
                    (
                        "Synthetic record unavailable".to_owned(),
                        ToolResultStatus::Error,
                    )
                } else {
                    (
                        "Synthetic evidence: score improved from 2 to 3.".to_owned(),
                        ToolResultStatus::Success,
                    )
                }
            }
            "write_section" => {
                let Document::Object(fields) = call.input() else {
                    panic!("object input");
                };
                let Document::String(text) = &fields["text"] else {
                    panic!("section text");
                };
                self.revisions.push(text.clone());
                if let Some(cancel) = &self.cancel_after_commit {
                    cancel.send(true).expect("cancel");
                }
                ("Revision committed".to_owned(), ToolResultStatus::Success)
            }
            _ => panic!("unoffered tool executed"),
        };
        Ok(ToolResultBlock::builder()
            .tool_use_id(call.tool_use_id())
            .content(ToolResultContentBlock::Text(text))
            .status(status)
            .build()
            .expect("result"))
    }
}

fn user(text: &str) -> Message {
    Message::builder()
        .role(ConversationRole::User)
        .content(ContentBlock::Text(text.to_owned()))
        .build()
        .expect("message")
}

fn call(id: &str, name: &str, text: &str) -> ContentBlock {
    ContentBlock::ToolUse(
        ToolUseBlock::builder()
            .tool_use_id(id)
            .name(name)
            .input(Document::Object(HashMap::from([(
                "text".to_owned(),
                Document::String(text.to_owned()),
            )])))
            .build()
            .expect("call"),
    )
}

fn response(content: Vec<ContentBlock>, stop_reason: StopReason) -> BedrockResponse {
    BedrockResponse {
        message: Message::builder()
            .role(ConversationRole::Assistant)
            .set_content(Some(content))
            .build()
            .expect("assistant"),
        stop_reason,
        usage: None,
        latency_ms: 7,
    }
}

fn request(text: &str) -> BedrockRequest<'static> {
    let tools = ["read_record", "write_section"]
        .into_iter()
        .map(|name| {
            Tool::ToolSpec(
                ToolSpecification::builder()
                    .name(name)
                    .description("Synthetic test capability")
                    .input_schema(ToolInputSchema::Json(Document::Object(HashMap::from([(
                        "type".to_owned(),
                        Document::String("object".to_owned()),
                    )]))))
                    .build()
                    .expect("tool"),
            )
        })
        .collect();
    BedrockRequest {
        model_id: "synthetic-model",
        system: Vec::new(),
        messages: vec![user(text)],
        tools: Some(
            ToolConfiguration::builder()
                .set_tools(Some(tools))
                .build()
                .expect("tools"),
        ),
        inference: InferenceConfiguration::builder().max_tokens(1024).build(),
        additional_fields: None,
        limits: StreamLimits::default(),
    }
}

fn limits() -> Limits {
    Limits {
        model_calls: NonZeroU32::new(8).expect("nonzero"),
        tool_calls: NonZeroU32::new(10).expect("nonzero"),
        messages: NonZeroUsize::new(32).expect("nonzero"),
        transcript_bytes: NonZeroUsize::new(65_536).expect("nonzero"),
    }
}

#[tokio::test]
async fn evidence_failure_repair_and_revision_preserve_native_replay_and_exact_output() {
    let reasoning = ContentBlock::ReasoningContent(ReasoningContentBlock::ReasoningText(
        ReasoningTextBlock::builder()
            .text("Synthetic opaque reasoning")
            .signature("synthetic-signature")
            .build()
            .expect("reasoning"),
    ));
    let expected = "## Findings\n\nScore: 2 → 3.\n\n| Visit | Score |\n| A | 2 |\n| B | 3 |";
    let model = ScriptedModel::new(vec![
        response(
            vec![reasoning.clone(), call("read-1", "read_record", "missing")],
            StopReason::ToolUse,
        ),
        response(
            vec![call("read-2", "read_record", "available")],
            StopReason::ToolUse,
        ),
        response(
            vec![call("write-1", "write_section", expected)],
            StopReason::ToolUse,
        ),
        response(
            vec![ContentBlock::Text(
                "Draft revised from the supplied evidence.".to_owned(),
            )],
            StopReason::EndTurn,
        ),
    ]);
    let mut host = DocumentHost::default();
    let mut visible = String::new();
    let outcome = agent::run(
        &model,
        &mut host,
        request("Revise the synthetic findings."),
        limits(),
        watch::channel(false).1,
        &mut |text| {
            visible.push_str(text);
            ControlFlow::Continue(())
        },
    )
    .await
    .expect("session");
    assert_eq!(host.revisions, [expected]);
    assert_eq!((outcome.model_calls, outcome.tool_calls), (4, 3));
    assert!(outcome.response.usage.is_none());
    assert_eq!(visible, "Draft revised from the supplied evidence.");
    let requests = model.requests.lock().expect("requests");
    assert_eq!(requests[1][1].content()[0], reasoning);
    let ContentBlock::ToolResult(failed) = &requests[1][2].content()[0] else {
        panic!("tool result");
    };
    assert_eq!(failed.status(), Some(&ToolResultStatus::Error));
    let ContentBlock::ToolResult(evidence) = &requests[2][4].content()[0] else {
        panic!("evidence");
    };
    assert_eq!(evidence.status(), Some(&ToolResultStatus::Success));
    for pair in requests.windows(2) {
        assert_eq!(pair[1][..pair[0].len()], pair[0]);
    }
}

#[tokio::test]
async fn unoffered_or_duplicate_calls_never_reach_the_host() {
    for (calls, failure) in [
        (
            vec![call("x", "send_email", "private")],
            ProtocolFailure::ToolNotOffered,
        ),
        (
            vec![
                call("x", "write_section", "first"),
                call("x", "write_section", "second"),
            ],
            ProtocolFailure::DuplicateToolId,
        ),
    ] {
        let model = ScriptedModel::new(vec![response(calls, StopReason::ToolUse)]);
        let mut host = DocumentHost::default();
        let error = agent::run(
            &model,
            &mut host,
            request("revise"),
            limits(),
            watch::channel(false).1,
            &mut |_| ControlFlow::Continue(()),
        )
        .await
        .err()
        .expect("refused");
        assert!(matches!(error, AgentError::Protocol(actual) if actual == failure));
        assert!(host.revisions.is_empty());
    }
}

#[tokio::test]
async fn a_tool_batch_over_budget_has_no_partial_effects() {
    let model = ScriptedModel::new(vec![response(
        vec![
            call("a", "write_section", "first"),
            call("b", "write_section", "second"),
        ],
        StopReason::ToolUse,
    )]);
    let mut host = DocumentHost::default();
    let mut bound = limits();
    bound.tool_calls = NonZeroU32::new(1).expect("nonzero");
    let error = agent::run(
        &model,
        &mut host,
        request("revise"),
        bound,
        watch::channel(false).1,
        &mut |_| ControlFlow::Continue(()),
    )
    .await
    .err()
    .expect("refused");
    assert!(matches!(error, AgentError::Limit(Resource::ToolCalls)));
    assert!(host.revisions.is_empty());
}

struct SilentModel(watch::Sender<bool>);

impl InferenceModel for SilentModel {
    async fn generate<'a>(
        &'a self,
        _: BedrockRequest<'a>,
        _: &'a mut (dyn FnMut(&str) -> ControlFlow<()> + Send),
    ) -> Result<BedrockResponse, BedrockError> {
        self.0.send(true).expect("started");
        std::future::pending().await
    }
}

#[tokio::test]
async fn silent_inference_can_stop_and_resume_with_the_users_latest_edit() {
    let (started, mut observed) = watch::channel(false);
    let (stop, cancel) = watch::channel(false);
    let model = SilentModel(started);
    let mut host = DocumentHost::default();
    let original = request("Original synthetic section");
    let original_messages = original.messages.clone();
    let mut ignore = |_: &str| ControlFlow::Continue(());
    let (result, ()) = tokio::join!(
        agent::run(&model, &mut host, original, limits(), cancel, &mut ignore),
        async {
            observed.wait_for(|value| *value).await.expect("started");
            stop.send(true).expect("stop");
        },
    );
    assert!(matches!(result, Err(AgentError::Cancelled)));
    assert!(host.revisions.is_empty());
    assert_eq!(original_messages, [user("Original synthetic section")]);
    let resumed = ScriptedModel::new(vec![
        response(
            vec![call(
                "revision",
                "write_section",
                "User-edited wording retained; evidence added.",
            )],
            StopReason::ToolUse,
        ),
        response(
            vec![ContentBlock::Text("Done".to_owned())],
            StopReason::EndTurn,
        ),
    ]);
    agent::run(
        &resumed,
        &mut host,
        request("User-edited wording retained"),
        limits(),
        watch::channel(false).1,
        &mut |_| ControlFlow::Continue(()),
    )
    .await
    .expect("resume");
    assert_eq!(
        host.revisions,
        ["User-edited wording retained; evidence added."]
    );
    assert_eq!(
        resumed.requests.lock().expect("requests")[0][0],
        user("User-edited wording retained")
    );
}

#[tokio::test]
async fn cancellation_does_not_drop_a_started_commit_and_prevents_the_next_effect() {
    let (stop, cancel) = watch::channel(false);
    let model = ScriptedModel::new(vec![response(
        vec![
            call("first", "write_section", "committed"),
            call("second", "write_section", "must not run"),
        ],
        StopReason::ToolUse,
    )]);
    let mut host = DocumentHost {
        cancel_after_commit: Some(stop),
        ..DocumentHost::default()
    };
    let result = agent::run(
        &model,
        &mut host,
        request("revise"),
        limits(),
        cancel,
        &mut |_| ControlFlow::Continue(()),
    )
    .await;
    assert!(matches!(result, Err(AgentError::Cancelled)));
    assert_eq!(host.revisions, ["committed"]);
}

#[tokio::test]
async fn truncation_is_not_a_successful_answer_but_complete_tool_calls_can_continue() {
    let mut host = DocumentHost::default();
    let truncated = ScriptedModel::new(vec![response(
        vec![ContentBlock::Text("incomplete".to_owned())],
        StopReason::MaxTokens,
    )]);
    let error = agent::run(
        &truncated,
        &mut host,
        request("revise"),
        limits(),
        watch::channel(false).1,
        &mut |_| ControlFlow::Continue(()),
    )
    .await
    .err()
    .expect("truncated");
    assert!(matches!(error, AgentError::Stopped(StopReason::MaxTokens)));
    let complete_call = ScriptedModel::new(vec![
        response(
            vec![call("write", "write_section", "complete section")],
            StopReason::MaxTokens,
        ),
        response(
            vec![ContentBlock::Text("Done".to_owned())],
            StopReason::EndTurn,
        ),
    ]);
    agent::run(
        &complete_call,
        &mut host,
        request("revise"),
        limits(),
        watch::channel(false).1,
        &mut |_| ControlFlow::Continue(()),
    )
    .await
    .expect("complete call salvage");
    assert_eq!(host.revisions, ["complete section"]);
}

#[tokio::test]
async fn transcript_over_budget_is_rejected_before_inference() {
    let model = ScriptedModel::new(vec![]);
    let mut host = DocumentHost::default();
    let mut bound = limits();
    bound.transcript_bytes = NonZeroUsize::new(4).expect("nonzero");
    let error = agent::run(
        &model,
        &mut host,
        request("too much context"),
        bound,
        watch::channel(false).1,
        &mut |_| ControlFlow::Continue(()),
    )
    .await
    .err()
    .expect("bounded");
    assert!(matches!(
        error,
        AgentError::Limit(Resource::TranscriptBytes)
    ));
    assert!(model.requests.lock().expect("requests").is_empty());
}

#[tokio::test]
async fn native_cache_points_survive_each_tool_round() {
    use dekopon_embedded::sdk::types::{CachePointBlock, CachePointType};
    let marker = ContentBlock::CachePoint(
        CachePointBlock::builder()
            .r#type(CachePointType::Default)
            .build()
            .expect("cache point"),
    );
    let mut input = request("Synthetic evidence");
    input.messages[0].content.push(marker.clone());
    let model = ScriptedModel::new(vec![
        response(
            vec![call("read", "read_record", "record")],
            StopReason::ToolUse,
        ),
        response(
            vec![ContentBlock::Text("Done".to_owned())],
            StopReason::EndTurn,
        ),
    ]);
    agent::run(
        &model,
        &mut DocumentHost::default(),
        input,
        limits(),
        watch::channel(false).1,
        &mut |_| ControlFlow::Continue(()),
    )
    .await
    .expect("cache replay");
    for request in model.requests.lock().expect("requests").iter() {
        assert_eq!(request[0].content()[1], marker);
    }
}

#[tokio::test]
async fn a_tool_id_reused_in_a_later_round_cannot_repeat_a_host_effect() {
    let model = ScriptedModel::new(vec![
        response(
            vec![call("same-id", "write_section", "first revision")],
            StopReason::ToolUse,
        ),
        response(
            vec![call("same-id", "write_section", "repeated effect")],
            StopReason::ToolUse,
        ),
    ]);
    let mut host = DocumentHost::default();
    let error = agent::run(
        &model,
        &mut host,
        request("revise"),
        limits(),
        watch::channel(false).1,
        &mut |_| ControlFlow::Continue(()),
    )
    .await
    .err()
    .expect("duplicate refused");
    assert!(matches!(
        error,
        AgentError::Protocol(ProtocolFailure::DuplicateToolId)
    ));
    assert_eq!(host.revisions, ["first revision"]);
}
