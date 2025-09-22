//! # Warning: Unstable API
//!
//! This library API is unstable and subject to breaking changes without notice.
//! Only the CLI interface is considered stable. Use at your own risk.
//!
//! To use the CLI tool, install with: `cargo install decon`

// All modules declared here for library structure
// Mark everything as doc(hidden) to avoid exposing unstable API in docs
#[doc(hidden)]
pub mod common;
#[doc(hidden)]
pub mod compare;
#[doc(hidden)]
pub mod detect;
#[doc(hidden)]
pub mod evals;
#[doc(hidden)]
pub mod review;
#[doc(hidden)]
pub mod server;

// Re-export execute functions for convenience (also hidden from docs)
#[doc(hidden)]
pub use compare::execute_compare;
#[doc(hidden)]
pub use detect::config::execute_detect;
#[doc(hidden)]
pub use evals::execute_evals;
#[doc(hidden)]
pub use review::execute_review;
#[doc(hidden)]
pub use server::config::execute_server;