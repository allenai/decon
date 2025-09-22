// Evaluation dataset management module
// This module provides functionality to download, analyze and display statistics for evaluation datasets.

pub mod args;
pub mod core;

// Re-export key types and functions for use in other modules
pub use args::EvalsArgs;
pub use core::{collect_eval_stats, execute_evals};