# Dekopon embedded in Claria

This package is a Claria-fork experiment, separate from Dekopon's native workspace.
It is not part of the upstream campaign or a published release. It preserves the
repository's MIT OR Apache-2.0 licensing and history. No Claria GPL source is
copied into this package.

The design follows the existing async inference interface in
`crates/dekopon-model/src/inference.rs`, complete-turn-before-dispatch ordering in
`crates/dekopon-agent/src/prompt.rs`, explicit cancellation in
`crates/dekopon-model/src/control.rs`, and separation of visible text from opaque
reasoning in `crates/dekopon-model/src/stream.rs`.

The host supplies native Bedrock messages and allowed tool definitions. The
backend accepts the host's AWS `SdkConfig`; it does not discover credentials,
provision infrastructure, or load another application's configuration. Streaming
returns a completed native assistant message, explicit stop reason and optional
unmodified token usage. The supported response profile is text, tool use and
signed/redacted reasoning; known image, native citation and server-tool-result
output is refused. Claria's citations remain inside its ordinary tool JSON.
Tool execution remains a host responsibility. Cancellation drops the in-flight
future and leaves no package-owned background task. Remote generation and billing
after disconnect require live validation.

Claria owns persistence, document/proposal validation, citations, authorization,
prompt construction, token budgets, cache placement, user edits and usage pricing.
The package introduces no history database, gateway, broker daemon, shell or
telemetry exporter. One `agent::drive` owns model-call and inference-invocation bounds,
retry waits and cancellation. The host selects retry eligibility and delays;
there is no retry loop inside the backend.

## Receipt and effect boundaries

`Session` adapts a domain response without translating it into a second transcript.
The driver calls `generate`, awaits `record` to completion, then calls `apply`.
It never drops a receipt write or admitted effect future when Stop arrives. A
completed response wins simultaneous cancellation so the host can record its usage.
By default Stop after a receipt prevents effects; a host can explicitly defer it
through an already accepted batch or its finalization cut. The native Bedrock
`run` adapter uses the same driver and checks cancellation between tools.

`model_failed` and `stopped(in_flight)` let the host retain unknown usage across
recovered retries visible to the driver. A missing provider receipt is not a
metered zero. No model retry includes effects from an incomplete response, and the driver never retries
host persistence or tools. Actual idempotency and conditional writes remain host
contracts.

The AWS SDK retains the host's retry configuration. Its internal attempts are
not additional driver invocations. A per-request signing-hook counter reports
`sdk_attempts`; when it exceeds one, the host must preserve final-response usage
but mark earlier usage unknown. Claria does this in both the immutable call
receipt and attempt totals. The counter observes SDK attempts, not provider
billing or every possible transport-level retry.

## Scoped fork contract

The native Dekopon constitution records model and tool payloads for its operator.
Claria explicitly requires private document content to stay out of diagnostic
traces. This package therefore emits only fixed event names, typed outcomes and
numeric counts; prompts, document text, reasoning, tool arguments/results,
credentials and patient identifiers must not be traced. Native daemon contracts
and behavior remain unchanged.

Claria's durable S3 sessions also remain outside this package. This is an embedding
boundary, not a new durability or provider-authorization implementation.

## Local checks

Run `cargo test --manifest-path embedded/Cargo.toml --locked` and
`cargo clippy --manifest-path embedded/Cargo.toml --all-targets --locked -- -D warnings`.
Claria integration tests exercise the real SDK against its loopback mock AWS
server. All fixtures contain synthetic content and dummy credentials. They prove
protocol and state-machine mechanics, not model quality or live AWS compatibility.

The byte bound applies to supported content after the SDK decodes each frame; it
does not bound SDK frame allocation or ignored future event payloads. Unknown
event handling still needs a wire fixture. Rust 1.97 is declared for Claria
compatibility; checks currently use the installed Rust 1.98.1.

Claria's sequential Writer and each planned section branch now implement `Session`.
They retain their distinct domain validation and retry policies while sharing this
driver. Claria saves immutable intermediate call receipts and resumes from its
existing durable document/run state, not a package-owned transcript database.
Orderly Stop/failure recovery is fixture-tested. Process-death takeover of a run
still marked active requires an explicit ownership design and is not provided here.

The inherited native dependency-audit policy does not pass for this package:
Claria's exact Tokio 1.52.3 pin has RUSTSEC-2026-0386, and AWS's SDK graph contains
both generations of `http` and `http-body`. No audit exceptions were added. A
coordinated dependency decision is required before this checkpoint can ship. An
isolated probe of checkpoint `94acc810f6e96cb6f48ec10065fdab5f0c8a788d` passes the
unchanged policy and all 23 checkpoint tests with Tokio 1.53.3, AWS config 1.12.0,
Bedrock Runtime 1.146.0 and Smithy types 1.8.0. The integrated pins are unchanged;
Claria's other exact AWS pins must move coherently before adopting that option.
