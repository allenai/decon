use anyhow::{Error, Result};
use crate::common::{Config, read_config, create_default_config, validate_config};
use crate::detect::args::DetectArgs;
use crate::detect::common_args::apply_common_detection_overrides;

pub fn execute_detect(args: &DetectArgs) -> Result<(), Error> {
    // Load config from file if provided, otherwise use defaults
    let mut loaded_config = if let Some(config_path) = &args.common.config {
        read_config(config_path)?
    } else {
        create_default_config()
    };

    apply_common_detection_overrides(&mut loaded_config, &args.common);

    // Ensure training_dir is provided
    if args.common.training_dir.is_none() && loaded_config.training_dir.as_os_str().is_empty() {
        return Err(anyhow::anyhow!(
            "training_dir is required. Provide it via --training-dir or in a config file\n\nThe training directory is a directory containing the training data to be decontaminated. It is expected to include jsonl files or jsonl.gz or jsonl.zst compressed files. The files are expected to have single text field containing text data. The name of the text field can be set with --content-key option (default \"text\").\n\nFor more details see:\n  decon detect --help\n\n"
        ));
    }

    validate_config(&loaded_config)?;

    // If purify is enabled but cleaned_output_dir is not set, generate a temp directory
    if loaded_config.purify && loaded_config.cleaned_output_dir.is_none() {
        loaded_config.cleaned_output_dir = Some(crate::common::generate_temp_dir("decon-cleaned"));
    }

    contamination_detect_with_config(&loaded_config)
}

fn contamination_detect_with_config(config_obj: &Config) -> Result<(), Error> {
    // Configure Rayon thread pool based on worker_threads setting
    rayon::ThreadPoolBuilder::new()
        .num_threads(config_obj.worker_threads)
        .build_global()
        .unwrap_or_else(|e| {
            eprintln!("Warning: Failed to set Rayon thread pool size: {}", e);
        });

    match config_obj.mode.as_str() {
        "simple" => {
            // Always announce the report output directory at the start
            let report_dir_str = config_obj.report_output_dir.display().to_string();
            print!("Contamination report output directory: {}", report_dir_str);

            // Check if this is an auto-generated temp directory and add hint
            if report_dir_str.contains("/tmp/decon-") || report_dir_str.contains("\\decon-") {
                println!(" (set report directory with --report-output-dir)");
            } else {
                println!();
            }

            // If purify is enabled, announce the cleaned output directory
            if config_obj.purify
                && let Some(ref cleaned_dir) = config_obj.cleaned_output_dir {
                    let cleaned_dir_str = cleaned_dir.display().to_string();
                    print!("Cleaned output directory: {}", cleaned_dir_str);

                    // Check if this is an auto-generated temp directory and add hint
                    if cleaned_dir_str.contains("/tmp/decon-cleaned-") || cleaned_dir_str.contains("\\decon-cleaned-") {
                        println!(" (set cleaned directory with --cleaned-output-dir)");
                    } else {
                        println!();
                    }
                }

            if config_obj.verbose {
                crate::vprintln!("Using Simple contamination detection...");
                crate::vprintln!("  N-gram size: {}", config_obj.ngram_size);
                crate::vprintln!(
                    "  Sample every M tokens: {}",
                    config_obj.sample_every_m_tokens
                );
                crate::vprintln!(
                    "  Question max consecutive misses: {}",
                    config_obj.question_max_consecutive_misses
                );
                crate::vprintln!("  Contamination score threshold: {}", config_obj.contamination_score_threshold);
                crate::vprintln!("  Tokenizer: {}", config_obj.tokenizer_str);
                crate::vprintln!("  Worker threads: {}", config_obj.worker_threads);

                // Display reference preprocessing options if any are enabled
                crate::vprintln!("\nReference Preprocessing:");
                crate::vprintln!(
                    "  Deduplication: {}",
                    if config_obj.eval_dedup {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
                if config_obj.eval_min_token_length > 0 {
                    crate::vprintln!(
                        "  Minimum length: {} tokens",
                        config_obj.eval_min_token_length
                    );
                }
                if config_obj.eval_min_unique_word_count > 0 {
                    crate::vprintln!(
                        "  Minimum unique words: {}",
                        config_obj.eval_min_unique_word_count
                    );
                }

                crate::vprintln!("\nInput and Output:");
                crate::vprintln!("  Training directory: {}", config_obj.training_dir.display());
                crate::vprintln!("  Content key: {}", config_obj.content_key);
                crate::vprintln!(
                    "  Evaluation directory: {}",
                    config_obj.evals_dir.display()
                );
                crate::vprintln!(
                    "  Report output dir: {}",
                    config_obj.report_output_dir.display()
                );
                if let Some(ref cleaned_dir) = config_obj.cleaned_output_dir {
                    crate::vprintln!("  Cleaned output dir: {}", cleaned_dir.display());
                }
                crate::vprintln!("  Purify: {}", config_obj.purify);
                crate::vprintln!();
            }

            super::contamination_detect(config_obj)
        }
        unknown_mode => {
            crate::vprintln!("Unknown mode: '{}'", unknown_mode);
            crate::vprintln!("Mode must be: simple");
            Err(anyhow::anyhow!(
                "Unsupported detection mode: {}",
                unknown_mode
            ))
        }
    }
}
