use anyhow::{Error, Result};
use crate::common::{read_config, create_default_config};
use crate::detect::common_args::apply_common_detection_overrides;
use super::args::ServerArgs;

/// Execute the server command with the given arguments
pub fn execute_server(args: &ServerArgs) -> Result<(), Error> {
    // Load config from file if provided, otherwise use defaults
    let mut loaded_config = if let Some(config_path) = &args.common.config {
        read_config(config_path)?
    } else {
        create_default_config()
    };

    // Apply common detection overrides
    apply_common_detection_overrides(&mut loaded_config, &args.common);

    // Server mode doesn't need training_dir since it processes individual files via API
    // Set a placeholder if not provided to avoid validation errors
    if loaded_config.training_dir.as_os_str().is_empty() {
        loaded_config.training_dir = std::path::PathBuf::from(".");
    }

    // Validate configuration for server mode (skip training_dir validation)
    validate_server_config(&loaded_config)?;

    // Run the server
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(super::run_server(loaded_config, args.port))
}

/// Validate configuration for server mode (doesn't require training_dir)
fn validate_server_config(config: &crate::common::Config) -> Result<(), Error> {
    // Validate evaluation directory exists and contains files
    if !config.evals_dir.exists() {
        return Err(anyhow::anyhow!(
            "Evaluation directory does not exist: {}\n\nPlease provide a valid directory containing evaluation datasets or run 'decon evals --download' to fetch the default evaluation datasets.",
            config.evals_dir.display()
        ));
    }

    if !config.evals_dir.is_dir() {
        return Err(anyhow::anyhow!(
            "Evaluation path is not a directory: {}\n\nPlease provide a directory containing evaluation datasets.",
            config.evals_dir.display()
        ));
    }

    // Check if evaluation directory contains any files (recursively)
    let has_eval_files = check_dir_for_data_files(&config.evals_dir);

    if !has_eval_files {
        return Err(anyhow::anyhow!(
            "Evaluation directory contains no data files: {}\n\nThe directory (including subdirectories) should contain JSON/JSONL files with supported extensions:\n  .json, .jsonl (uncompressed or with .gz, .zst, .zstd, .bz2, .xz compression)\n\nYou can download the default evaluation datasets by running: decon evals --download",
            config.evals_dir.display()
        ));
    }

    // Validate report output directory can be created
    if let Some(parent) = config.report_output_dir.parent()
        && !parent.exists() {
            std::fs::create_dir_all(parent).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to create report output directory parent: {}\nError: {}",
                    parent.display(),
                    e
                )
            })?;
        }

    // Validate cleaned output directory can be created if purify is enabled
    if config.purify
        && let Some(ref cleaned_dir) = config.cleaned_output_dir
            && let Some(parent) = cleaned_dir.parent()
                && !parent.exists() {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        anyhow::anyhow!(
                            "Failed to create cleaned output directory parent: {}\nError: {}",
                            parent.display(),
                            e
                        )
                    })?;
                }

    Ok(())
}

/// Helper function to recursively check if a directory contains supported data files
fn check_dir_for_data_files(dir: &std::path::PathBuf) -> bool {
    fn check_dir_recursive(dir: &std::path::PathBuf) -> bool {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.filter_map(Result::ok) {
                let path = entry.path();
                if path.is_file() {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        // Check if the file has any of the supported extensions
                        for ext in crate::common::SUPPORTED_DATA_EXTENSIONS {
                            if name.ends_with(ext) {
                                return true;
                            }
                        }
                    }
                } else if path.is_dir() {
                    // Recursively check subdirectories
                    if check_dir_recursive(&path) {
                        return true;
                    }
                }
            }
        }
        false
    }
    check_dir_recursive(dir)
}
