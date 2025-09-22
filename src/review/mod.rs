pub mod args;
pub mod display;
mod stats;
mod counts;
mod top_examples;
mod filters;
mod loader;
mod types;
mod core;

use args::ReviewArgs;
use anyhow::{Error, Result};

// Re-export public types and functions for other modules to use
pub use types::{ContaminationResult, ContaminationResultWithSource};
pub use filters::filter_contamination_results_by_thresholds;
pub use loader::load_contamination_results_from_directory;
pub use core::review_contamination;

/// Execute the review command
pub fn execute_review(args: &ReviewArgs) -> Result<(), Error> {
    review_contamination(args)
}