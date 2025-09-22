// Module declarations
// Public API modules - exposed to external users
pub mod args;
pub mod common_args;
pub mod config;
pub mod contamination_entry;
pub mod detection;
pub mod scoring;
pub mod reporting;
pub mod reference_index;

// Crate-internal modules - used within the crate
pub(crate) mod stats;
pub(crate) mod cluster;
pub(crate) mod hot_bucket_stats;

// Private implementation modules
mod answer_boundary;
mod passage_boundary;
mod utils;
pub(crate) mod display;

// High-level convenience re-exports for the most commonly used items
// These provide a simpler API without requiring users to navigate submodules
pub use config::execute_detect;
pub use contamination_entry::SimpleContaminationEntry;
pub use detection::{contamination_detect, ContaminationResults};
pub use reference_index::SimpleReferenceIndex;