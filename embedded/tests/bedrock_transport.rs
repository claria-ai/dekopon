#![allow(
    clippy::disallowed_methods,
    reason = "tests own and join loopback peers"
)]

use std::{ops::ControlFlow, time::Duration};

use dekopon_embedded::{
    bedrock::{
        BedrockClient, BedrockError, BedrockRequest, FailurePhase, InterruptionKind, StreamLimits,
    },
    sdk::{
        config::{Credentials, Region, SharedCredentialsProvider},
        types::{ContentBlock, ConversationRole, InferenceConfiguration, Message},
    },
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    sync::oneshot,
    task::JoinHandle,
};

async fn silent_peer() -> (String, oneshot::Receiver<()>, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("loopback listener");
    let endpoint = format!(
        "http://{}",
        listener.local_addr().expect("loopback address")
    );
    let (sent, ready) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.expect("one request");
        let mut request = Vec::new();
        loop {
            let mut bytes = [0; 1024];
            let count = stream.read(&mut bytes).await.expect("request bytes");
            assert_ne!(count, 0);
            request.extend_from_slice(&bytes[..count]);
            assert!(request.len() < 16 * 1024);
            if let Some(end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
                let headers = std::str::from_utf8(&request[..end]).expect("request headers");
                let length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().expect("body length"))
                    })
                    .expect("content length");
                if request.len() >= end + 4 + length {
                    break;
                }
            }
        }
        assert!(request.starts_with(b"POST /model/synthetic-model/converse-stream "));
        stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/vnd.amazon.eventstream\r\nTransfer-Encoding: chunked\r\n\r\n8\r\n\0\0\0\0\0\0\0\0\r\n").await.expect("headers and incomplete frame");
        sent.send(()).expect("reader waiting");
        std::future::pending::<()>().await;
    });
    (endpoint, ready, server)
}

fn client(endpoint: &str) -> BedrockClient {
    BedrockClient::new(
        &aws_config::SdkConfig::builder()
            .region(Region::new("us-east-1"))
            .credentials_provider(SharedCredentialsProvider::new(Credentials::new(
                "test",
                "test",
                None,
                None,
                "synthetic",
            )))
            .endpoint_url(endpoint)
            .behavior_version(aws_config::BehaviorVersion::latest())
            .build(),
    )
}

fn request() -> BedrockRequest<'static> {
    BedrockRequest {
        model_id: "synthetic-model",
        system: Vec::new(),
        tools: None,
        additional_fields: None,
        messages: vec![
            Message::builder()
                .role(ConversationRole::User)
                .content(ContentBlock::Text("Synthetic prompt".into()))
                .build()
                .expect("fixture message"),
        ],
        inference: InferenceConfiguration::builder().max_tokens(32).build(),
        limits: StreamLimits::new(Duration::from_secs(3), Duration::from_secs(30), 1024, 8, 2)
            .expect("limits"),
    }
}

#[tokio::test]
async fn headers_do_not_end_the_first_frame_deadline() {
    let (endpoint, ready, server) = silent_peer().await;
    let client = client(&endpoint);
    let started = tokio::time::Instant::now();
    let mut observe = |_: &str| ControlFlow::Continue(());
    let (result, ()) = tokio::join!(client.converse(request(), &mut observe), async {
        ready.await.expect("headers sent");
        tokio::time::pause();
    });
    server.abort();
    assert!(server.await.expect_err("peer stopped").is_cancelled());
    assert!(matches!(
        result,
        Err(BedrockError::Interrupted {
            phase: FailurePhase::AwaitingResponse,
            kind: InterruptionKind::Timeout
        })
    ));
    assert!(started.elapsed() >= Duration::from_secs(3));
    assert!(started.elapsed() < Duration::from_secs(30));
}

#[tokio::test]
async fn a_silent_first_frame_wait_can_be_cancelled_by_dropping_the_exchange() {
    let (endpoint, ready, server) = silent_peer().await;
    let client = client(&endpoint);
    let mut observe = |_: &str| ControlFlow::Continue(());
    tokio::select! {
        result = client.converse(request(), &mut observe) => {
            let _ = result;
            panic!("the silent response cannot finish before cancellation");
        }
        result = ready => result.expect("headers arrived before cancellation"),
    }
    server.abort();
    assert!(server.await.expect_err("peer stopped").is_cancelled());
}
