pub mod args;
pub mod core;
pub mod display;
pub mod metrics;

// Re-export key types and functions for use in other modules
pub use args::CompareArgs;
pub use core::{execute_compare, ContaminationKey};
pub use display::{display_common, display_comparison_stats, display_only_in_first, display_only_in_second, ComparisonData};
pub use metrics::compute_contamination_metrics;