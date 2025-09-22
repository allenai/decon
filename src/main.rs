// External crates
use anyhow::{Error, Result};
use clap::{Parser, Subcommand};

// Import from library crate
use decon::{
    compare, detect, evals, review, server,
    execute_compare, execute_detect, execute_evals, execute_review, execute_server,
};


#[derive(Parser)]
#[clap(
    author,
    version,
    about = "Decon - A contamination detection tool for machine learning datasets",
    long_about = "Decon identifies when training data contains text from evaluation datasets.\n\nUses the SIMPLE detection algorithm:\n- N-gram matching with sampling and cluster expansion\n- Configurable thresholds for questions and answers\n- Support for multiple tokenizers (r50k, p50k, p50k_edit, cl100k, o200k, uniseg)\n\nConfiguration options can be specified in a YAML file and overridden via command-line flags.",
    disable_help_subcommand = true
)]
struct ArgParser {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Detect(detect::args::DetectArgs),
    Server(server::args::ServerArgs),

    Review(review::args::ReviewArgs),

    Compare(compare::args::CompareArgs),

    Evals(evals::args::EvalsArgs),
}

fn main() -> Result<(), Error> {
    let args = ArgParser::parse();

    
    match &args.command {
        Commands::Detect(args) => execute_detect(args),

        Commands::Review(args) => execute_review(args),

        Commands::Server(args) => execute_server(args),

        Commands::Evals(args) => execute_evals(args),

        Commands::Compare(args) => execute_compare(args),
    }
}
