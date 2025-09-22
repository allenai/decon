use clap::Args;
use crate::detect::common_args::CommonDetectionArgs;

#[derive(Args, Debug)]
#[command(about = "Run decon as an HTTP server for orchestrated pipelines")]
pub struct ServerArgs {
    // Server Configuration
    #[arg(
        long,
        default_value_t = 8080,
        help = "Port to listen on",
        help_heading = "Server Configuration"
    )]
    pub port: u16,

    // All the common detection arguments
    #[command(flatten)]
    pub common: CommonDetectionArgs,
}