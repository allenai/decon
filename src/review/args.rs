use clap::Args;
use std::path::PathBuf;

#[derive(Args, Debug)]
#[command(
    about = "Review and analyze detect results",
    long_about = "Review contamination detection results with various display modes and filtering options.\nAllows interactive stepping through results or batch analysis with statistics.",
    after_help = "Examples:\n    decon review /tmp/decon-report                      # Interactive review mode\n    decon review /tmp/decon-report --stats              # Show dataset statistics\n    decon review /tmp/decon-report --min-score 0.9      # Filter high-confidence matches"
)]
pub struct ReviewArgs {
    #[arg(
        help = "Directory containing result files to analyze",
        display_order = 1
    )]
    pub dir: PathBuf,

    // Display Modes
    #[arg(
        long,
        help = "Display eval dataset statistics with horizontal bar chart",
        display_order = 10,
        help_heading = "Display Modes"
    )]
    pub stats: bool,

    #[arg(
        long,
        help = "Dump all contamination results at once (without stepping through)",
        display_order = 11,
        help_heading = "Display Modes"
    )]
    pub dump: bool,

    #[arg(
        long,
        help = "Show top N most commonly matched eval examples",
        value_name = "N",
        display_order = 12,
        help_heading = "Display Modes"
    )]
    pub top_eval_examples: Option<usize>,

    #[arg(
        long,
        help = "Display counts of unique training documents per dataset that need removal",
        display_order = 13,
        help_heading = "Display Modes"
    )]
    pub dataset_counts: bool,

    // Filtering
    #[arg(
        long,
        help = "Minimum contamination score to include in results",
        display_order = 20,
        help_heading = "Filtering"
    )]
    pub min_score: Option<f32>,

    #[arg(
        long,
        help = "Minimum n-gram match count to include in results (filters by prompt ngram_match_cnt)",
        display_order = 21,
        help_heading = "Filtering"
    )]
    pub min_length: Option<usize>,

    #[arg(
        long,
        help = "Filter by evaluation dataset name",
        display_order = 22,
        help_heading = "Filtering"
    )]
    pub eval: Option<String>,

    // Sorting
    #[arg(
        long,
        help = "Sort results by n-gram match count in descending order (highest matches first)",
        display_order = 30,
        help_heading = "Sorting"
    )]
    pub sort_match_length_descending: bool,

    #[arg(
        long,
        help = "Sort results by n-gram match count in ascending order (lowest matches first)",
        display_order = 31,
        help_heading = "Sorting"
    )]
    pub sort_match_length_ascending: bool,

    // Output
    #[arg(
        long,
        short = 'v',
        help = "Show verbose output with all scores and metrics",
        display_order = 40,
        help_heading = "Output"
    )]
    pub verbose: bool,
}
