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
telemetry exporter. The host owns retries so nested loops cannot multiply them.

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

The initial checkpoint integrates only the backend into Claria's existing Writer
calls. The small tool-loop kernel is tested separately pending review before
migrating Claria's sequential and parallel executors. It does not yet expose
intermediate call receipts or restore a transcript after a committed multi-step
interruption. Those are requirements for production migration, still owned by
Claria's existing loop. Its synthetic resume test starts fresh with edited context.

The inherited native dependency-audit policy does not pass for this package:
Claria's exact Tokio 1.52.3 pin has RUSTSEC-2026-0386, and AWS's SDK graph contains
both generations of `http` and `http-body`. No audit exceptions were added. A
coordinated dependency decision is required before this checkpoint can ship.
