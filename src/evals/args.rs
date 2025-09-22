use clap::Args;
use std::path::PathBuf;

#[derive(Args, Debug)]
#[command(
    about = "Manage evaluation datasets to decontaminate against",
    long_about = "List available evaluation datasets or download new ones.\nBy default (no options), displays statistics for all available evaluation datasets.\n\nDecon uses a directory of reference files for decontamination. Default: ~/.local/share/decon/references\n(if downloaded) or bundled-evals. Override with --evals-dir when running detect.\n\nCurate your own eval set with this command's tools. Use --download with a YAML config to fetch\nHuggingFace datasets (set HF environment variables for restricted access). Or prepare files manually\nusing the format shown in bundled-evals/."
)]
pub struct EvalsArgs {
    /// Override directory containing evaluation datasets to analyze
    #[arg(
        long,
        help = "Directory containing evaluation datasets to analyze",
        value_name = "DIR",
        help_heading = "Listing Options"
    )]
    pub dir: Option<PathBuf>,

    /// Show detailed statistics (min/avg/max lengths) for questions, answers, and passages
    #[arg(
        long,
        help = "Show detailed statistics including min/avg/max lengths",
        help_heading = "Listing Options"
    )]
    pub stats: bool,

    /// Download all evaluation datasets
    #[arg(
        long,
        help = "Download all evaluation datasets from HuggingFace",
        help_heading = "Download Options"
    )]
    pub download: bool,

    /// Download a specific evaluation dataset by name
    #[arg(
        long,
        help = "Download a specific evaluation dataset by name",
        value_name = "NAME",
        conflicts_with = "download",
        help_heading = "Download Options"
    )]
    pub eval: Option<String>,

    /// Output directory for downloaded datasets
    #[arg(
        long,
        help = "Output directory for downloaded datasets (default: ~/.local/share/decon/references)",
        value_name = "DIR",
        help_heading = "Download Options"
    )]
    pub output_dir: Option<PathBuf>,

    /// Configuration file for evaluation datasets
    #[arg(
        long,
        help = "Configuration file for evaluation datasets (default: ~/.local/share/decon/evals.yaml)",
        value_name = "PATH",
        help_heading = "Download Options"
    )]
    pub config: Option<PathBuf>,
}
