use clap::Args;
use crate::detect::common_args::CommonDetectionArgs;

#[derive(Args, Debug)]
#[command(
    about = "Detect contamination in training data",
    long_about = "Detect and report on contamination between a reference dataset and a training dataset.\nCan optionally produce a clean dataset with contaminated documents removed.",
    after_help = "Example:\n    decon detect --training-dir /tmp/my-directory-of-jsonl-files --report-output-dir /tmp/decon-report"
)]
pub struct DetectArgs {
    #[command(flatten)]
    pub common: CommonDetectionArgs,
}