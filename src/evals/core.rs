use super::args::EvalsArgs;
use anyhow::Error;
use flate2::read::GzDecoder;
use mj_io::expand_dirs;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Execute the evals command
pub fn execute_evals(args: &EvalsArgs) -> Result<(), Error> {
    if args.download {
        download_evals(None, args.output_dir.as_deref(), args.config.as_deref())
    } else if let Some(eval_name) = &args.eval {
        download_evals(Some(eval_name.clone()), args.output_dir.as_deref(), args.config.as_deref())
    } else {
        // Default behavior: show stats for evaluation datasets
        // Resolve which directory to use
        let evals_dir = if let Some(dir) = &args.dir {
            // User explicitly provided a directory
            eprintln!("Using evaluation directory from --dir: {}\n", dir.display());
            dir.clone()
        } else {
            // Check for user-downloaded evals
            let user_evals_dir = dirs::data_local_dir()
                .map(|d| d.join("decon").join("references"))
                .unwrap_or_else(|| PathBuf::from("~/.local/share/decon/references"));

            if user_evals_dir.exists() && user_evals_dir.is_dir() {
                let has_files = std::fs::read_dir(&user_evals_dir)
                    .map(|entries| {
                        entries.filter_map(Result::ok).any(|entry| {
                            entry.path().extension()
                                .and_then(|ext| ext.to_str())
                                .map(|ext| ext == "jsonl" || ext == "gz")
                                .unwrap_or(false)
                        })
                    })
                    .unwrap_or(false);

                if has_files {
                    eprintln!("Using user-downloaded evaluation directory: {}\n", user_evals_dir.display());
                    user_evals_dir
                } else {
                    let default_dir = PathBuf::from("bundled-evals");
                    eprintln!("Using default evaluation directory: {}\n", default_dir.display());
                    default_dir
                }
            } else {
                let default_dir = PathBuf::from("bundled-evals");
                eprintln!("Using default evaluation directory: {}\n", default_dir.display());
                default_dir
            }
        };

        collect_eval_stats(&evals_dir, args.stats)
    }
}

/// Download evaluation datasets using the Python evals.py script
fn download_evals(eval_name: Option<String>, output_dir: Option<&Path>, config_path: Option<&Path>) -> Result<(), Error> {
    // Determine config file path
    let default_config_dir = dirs::data_local_dir()
        .map(|d| d.join("decon"))
        .unwrap_or_else(|| PathBuf::from("~/.local/share/decon"));

    let config_file = if let Some(config) = config_path {
        // User provided a config path
        config.to_path_buf()
    } else {
        // Use default location
        default_config_dir.join("evals.yaml")
    };

    // If config doesn't exist and we're using the default location, copy from bundled
    if config_path.is_none() && !config_file.exists() {
        // Ensure directory exists
        if let Some(parent) = config_file.parent() {
            std::fs::create_dir_all(parent).ok();
        }

        // Copy default config from the bundled location
        let bundled_config = PathBuf::from("config/evals.yaml");
        if bundled_config.exists() {
            if let Err(e) = std::fs::copy(&bundled_config, &config_file) {
                eprintln!("Note: Could not copy default config to {}: {}", config_file.display(), e);
                eprintln!("Will use bundled config from {}", bundled_config.display());
            } else {
                println!("✓ Created user configuration file at: {}", config_file.display());
                println!("  You can edit this file to customize which evaluation datasets to download.");
            }
        }
    }

    // Use the config file if it exists, otherwise fall back to bundled
    let final_config = if config_file.exists() {
        println!("Using configuration file: {}", config_file.display());
        config_file
    } else if config_path.is_some() {
        // User specified a config that doesn't exist
        return Err(anyhow::anyhow!("Config file not found: {}", config_file.display()));
    } else {
        // Fall back to bundled config
        let bundled = PathBuf::from("config/evals.yaml");
        println!("Using bundled configuration file: {}", bundled.display());
        bundled
    };
    println!("Tip: Edit the configuration file to customize which evaluation datasets to download.");
    println!();
    let python_check = Command::new("python3")
        .arg("--version")
        .output()
        .or_else(|_| Command::new("python").arg("--version").output());

    if python_check.is_err() {
        return Err(anyhow::anyhow!(
            "Python is not available. Please install Python 3 to use the download feature."
        ));
    }

    // Determine Python command (prefer python3)
    let python_cmd = if Command::new("python3").arg("--version").output().is_ok() {
        "python3"
    } else {
        "python"
    };

    let script_path = "python/evals.py";
    let mut cmd = Command::new(python_cmd);
    cmd.arg(script_path);
    cmd.arg("--config").arg(&final_config);

    if let Some(eval) = eval_name {
        println!("Downloading evaluation dataset: {}", eval);
        cmd.arg("--eval").arg(eval);
    } else {
        println!("Downloading all evaluation datasets...");
        println!("This will download datasets from HuggingFace and may take some time.");
        cmd.arg("--download");
    }

    // Add output directory if specified
    if let Some(dir) = output_dir {
        cmd.arg("--output-dir").arg(dir);
        println!("Output directory: {:?}", dir);
    } else {
        let default_dir = dirs::data_local_dir()
            .map(|d| d.join("decon").join("references"))
            .unwrap_or_else(|| PathBuf::from("~/.local/share/decon/references"));
        println!("Output directory: {:?} (default)", default_dir.display());
    }

    println!("Starting download process...");
    println!();

    let status = cmd
        .status()
        .map_err(|e| anyhow::anyhow!("Failed to execute Python script: {}", e))?;

    if !status.success() {
        let exit_code = status.code().unwrap_or(-1);
        return Err(anyhow::anyhow!(
            "Python script failed with exit code: {}",
            exit_code
        ));
    }

    Ok(())
}

pub fn collect_eval_stats(stats_dir: &Path, show_stats: bool) -> Result<(), Error> {
    if !stats_dir.exists() {
        eprintln!("Error: Directory not found: {}", stats_dir.display());
        return Err(anyhow::anyhow!("Directory not found: {}", stats_dir.display()));
    }

    // Find all JSONL files in the directory
    let eval_files = expand_dirs(
        vec![stats_dir.to_path_buf()],
        Some(vec![".jsonl", ".gz"].as_slice()),
    )?;

    if eval_files.is_empty() {
        println!("No JSONL files found in {}", stats_dir.display());
        return Ok(());
    }

    // Collect stats for each eval dataset
    let mut eval_stats: HashMap<String, EvalStats> = HashMap::new();
    let mut total_skipped_records = 0usize;

    for file_path in &eval_files {
        match process_file_for_stats(file_path, &mut eval_stats) {
            Ok(skipped_count) => {
                total_skipped_records += skipped_count;
            }
            Err(e) => {
                eprintln!("Error processing file {:?}: {:?}", file_path, e);
            }
        }
    }

    // Display the stats in table format
    display_stats_table(&eval_stats, total_skipped_records, show_stats);

    Ok(())
}

#[derive(Default)]
struct EvalStats {
    question_count: usize,
    answer_count: usize,
    passage_count: usize,
    question_lengths: Vec<usize>,
    answer_lengths: Vec<usize>,
    passage_lengths: Vec<usize>,
}

fn process_file_for_stats(file_path: &PathBuf, eval_stats: &mut HashMap<String, EvalStats>) -> Result<usize, Error> {
    let file = File::open(file_path)?;
    let reader: Box<dyn BufRead> = if file_path.extension().and_then(|s| s.to_str()) == Some("gz") {
        Box::new(BufReader::new(GzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    let mut skipped_records = 0usize;

    for line in reader.lines() {
        let line = line?;

        if line.trim().is_empty() {
            continue;
        }

        let json_obj: Value = match serde_json::from_str(&line) {
            Ok(obj) => obj,
            Err(_) => continue, // Skip invalid JSON
        };

        let eval_key = match json_obj.get("eval_key").and_then(|v| v.as_str()) {
            Some(key) if !key.trim().is_empty() => key.to_string(),
            _ => {
                skipped_records += 1;
                continue; // Skip records without eval_key
            }
        };

        let stats = eval_stats
            .entry(eval_key)
            .or_default();

        if let Some(question) = json_obj.get("question").and_then(|v| v.as_str())
            && !question.trim().is_empty() {
                stats.question_count += 1;
                stats.question_lengths.push(question.len());
            }

        if let Some(answer) = json_obj.get("answer").and_then(|v| v.as_str())
            && !answer.trim().is_empty() {
                stats.answer_count += 1;
                stats.answer_lengths.push(answer.len());
            }

        if let Some(passage) = json_obj.get("passage").and_then(|v| v.as_str())
            && !passage.trim().is_empty() {
                stats.passage_count += 1;
                stats.passage_lengths.push(passage.len());
            }
    }

    Ok(skipped_records)
}

fn display_stats_table(eval_stats: &HashMap<String, EvalStats>, skipped_records: usize, show_stats: bool) {
    println!("Reference Dataset Statistics");
    println!("============================");
    if show_stats {
        println!("Note: Q = Questions, A = Answers, P = Passages. Length statistics (Min/Avg/Max) are measured in characters, not words or tokens.");
    } else {
        println!("Note: Q = Questions, A = Answers, P = Passages. Use --stats flag to show detailed length statistics.");
    }
    println!();

    // Sort alphabetically by eval name
    let mut sorted_evals: Vec<(&String, &EvalStats)> = eval_stats.iter().collect();
    sorted_evals.sort_by(|a, b| a.0.cmp(b.0));

    // Calculate column width based on longest eval name
    let eval_width = eval_stats
        .keys()
        .map(|name| name.len())
        .max()
        .unwrap_or(20)
        .max(9); // At least 9 chars for "Eval Name" header

    if show_stats {
        // Print full table header with statistics
        println!(
            "┌{:─<width$}┬{:─>11}┬{:─>11}┬{:─>11}┬{:─>9}┬{:─>9}┬{:─>9}┬{:─>9}┬{:─>9}┬{:─>9}┬{:─>9}┬{:─>9}┬{:─>9}┐",
            "─",
            "─",
            "─",
            "─",
            "─",
            "─",
            "─",
            "─",
            "─",
            "─",
            "─",
            "─",
            "─",
            width = eval_width + 2
        );
        println!(
            "│ {:<width$} │ {:>9} │ {:>9} │ {:>9} │ {:>7} │ {:>7} │ {:>7} │ {:>7} │ {:>7} │ {:>7} │ {:>7} │ {:>7} │ {:>7} │",
            "Eval Name",
            "Q",
            "A",
            "P",
            "Min Q",
            "Avg Q",
            "Max Q",
            "Min A",
            "Avg A",
            "Max A",
            "Min P",
            "Avg P",
            "Max P",
            width = eval_width
        );
        println!(
            "├{:─<width$}┼{:─>11}┼{:─>11}┼{:─>11}┼{:─>9}┼{:─>9}┼{:─>9}┼{:─>9}┼{:─>9}┼{:─>9}┼{:─>9}┼{:─>9}┼{:─>9}┤",
            "─",
            "─",
            "─",
            "─",
            "─",
            "─",
            "─",
            "─",
            "─",
            "─",
            "─",
            "─",
            "─",
            width = eval_width + 2
        );
    } else {
        // Print simple table header without statistics
        println!(
            "┌{:─<width$}┬{:─>11}┬{:─>11}┬{:─>11}┐",
            "─",
            "─",
            "─",
            "─",
            width = eval_width + 2
        );
        println!(
            "│ {:<width$} │ {:>9} │ {:>9} │ {:>9} │",
            "Eval Name",
            "Q",
            "A",
            "P",
            width = eval_width
        );
        println!(
            "├{:─<width$}┼{:─>11}┼{:─>11}┼{:─>11}┤",
            "─",
            "─",
            "─",
            "─",
            width = eval_width + 2
        );
    }

    // Print each eval's stats
    for (eval_name, stats) in sorted_evals {
        if show_stats {
            // Calculate statistics for full display
            let (min_q, avg_q, max_q) = if !stats.question_lengths.is_empty() {
                let min = *stats.question_lengths.iter().min().expect("Question lengths should have min value when non-empty");
                let avg = stats.question_lengths.iter().sum::<usize>() as f64
                    / stats.question_lengths.len() as f64;
                let max = *stats.question_lengths.iter().max().expect("Question lengths should have max value when non-empty");
                (min.to_string(), format!("{:.0}", avg), max.to_string())
            } else {
                ("-".to_string(), "-".to_string(), "-".to_string())
            };

            let (min_a, avg_a, max_a) = if !stats.answer_lengths.is_empty() {
                let min = *stats.answer_lengths.iter().min().expect("Answer lengths should have min value when non-empty");
                let avg = stats.answer_lengths.iter().sum::<usize>() as f64
                    / stats.answer_lengths.len() as f64;
                let max = *stats.answer_lengths.iter().max().expect("Answer lengths should have max value when non-empty");
                (min.to_string(), format!("{:.0}", avg), max.to_string())
            } else {
                ("-".to_string(), "-".to_string(), "-".to_string())
            };

            let (min_p, avg_p, max_p) = if !stats.passage_lengths.is_empty() {
                let min = *stats.passage_lengths.iter().min().expect("Passage lengths should have min value when non-empty");
                let avg = stats.passage_lengths.iter().sum::<usize>() as f64
                    / stats.passage_lengths.len() as f64;
                let max = *stats.passage_lengths.iter().max().expect("Passage lengths should have max value when non-empty");
                (min.to_string(), format!("{:.0}", avg), max.to_string())
            } else {
                ("-".to_string(), "-".to_string(), "-".to_string())
            };

            println!(
                "│ {:<width$} │ {:>9} │ {:>9} │ {:>9} │ {:>7} │ {:>7} │ {:>7} │ {:>7} │ {:>7} │ {:>7} │ {:>7} │ {:>7} │ {:>7} │",
                eval_name,
                stats.question_count,
                stats.answer_count,
                stats.passage_count,
                min_q,
                avg_q,
                max_q,
                min_a,
                avg_a,
                max_a,
                min_p,
                avg_p,
                max_p,
                width = eval_width
            );
        } else {
            // Simple display without statistics
            println!(
                "│ {:<width$} │ {:>9} │ {:>9} │ {:>9} │",
                eval_name,
                stats.question_count,
                stats.answer_count,
                stats.passage_count,
                width = eval_width
            );
        }
    }

    // Print table footer
    if show_stats {
        println!(
            "└{:─<width$}┴{:─>11}┴{:─>11}┴{:─>11}┴{:─>9}┴{:─>9}┴{:─>9}┴{:─>9}┴{:─>9}┴{:─>9}┴{:─>9}┴{:─>9}┴{:─>9}┘",
            "─",
            "─",
            "─",
            "─",
            "─",
            "─",
            "─",
            "─",
            "─",
            "─",
            "─",
            "─",
            "─",
            width = eval_width + 2
        );
    } else {
        println!(
            "└{:─<width$}┴{:─>11}┴{:─>11}┴{:─>11}┘",
            "─",
            "─",
            "─",
            "─",
            width = eval_width + 2
        );
    }

    // Print summary
    let total_questions: usize = eval_stats.values().map(|s| s.question_count).sum();
    let total_answers: usize = eval_stats.values().map(|s| s.answer_count).sum();
    let total_passages: usize = eval_stats.values().map(|s| s.passage_count).sum();
    println!("\nTotal evals: {}", eval_stats.len());
    println!("Total questions: {}", total_questions);
    println!("Total answers: {}", total_answers);
    println!("Total passages: {}", total_passages);

    // Display warning if any records were skipped
    if skipped_records > 0 {
        println!("\nWarning: Skipped {} records without eval_key field", skipped_records);
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_eval_stats_default() {
        let stats = EvalStats::default();
        assert_eq!(stats.question_count, 0);
        assert_eq!(stats.answer_count, 0);
        assert_eq!(stats.passage_count, 0);
        assert!(stats.question_lengths.is_empty());
        assert!(stats.answer_lengths.is_empty());
        assert!(stats.passage_lengths.is_empty());
    }

    #[test]
    fn test_process_file_with_valid_json() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_eval.jsonl");

        // Create test file with valid JSON including eval_key
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, r#"{{"eval_key": "test_eval", "question": "What is 2+2?", "answer": "4"}}"#).unwrap();
        writeln!(file, r#"{{"eval_key": "test_eval", "question": "What is the capital of France?", "answer": "Paris", "passage": "France is a country in Europe."}}"#).unwrap();
        writeln!(file, r#"{{"eval_key": "test_eval", "question": "", "answer": "Empty question"}}"#).unwrap(); // Empty question should be skipped
        writeln!(file, r#"{{"question": "Missing eval_key", "answer": "Should be skipped"}}"#).unwrap(); // Should be skipped

        let mut eval_stats = HashMap::new();
        let skipped = process_file_for_stats(&file_path, &mut eval_stats).unwrap();

        assert_eq!(eval_stats.len(), 1);
        assert_eq!(skipped, 1); // One record without eval_key
        let stats = eval_stats.get("test_eval").unwrap();
        assert_eq!(stats.question_count, 2);
        assert_eq!(stats.answer_count, 3);
        assert_eq!(stats.passage_count, 1);

        // Check lengths
        assert_eq!(stats.question_lengths.len(), 2);
        assert_eq!(stats.answer_lengths.len(), 3);
        assert_eq!(stats.passage_lengths.len(), 1);
    }

    #[test]
    fn test_process_file_with_invalid_json() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_eval.jsonl");

        // Create test file with some invalid JSON lines
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, r#"{{"eval_key": "test_eval", "question": "Valid JSON", "answer": "yes"}}"#).unwrap();
        writeln!(file, "This is not JSON").unwrap(); // Should be skipped
        writeln!(file).unwrap(); // Empty line should be skipped
        writeln!(file, r#"{{"eval_key": "test_eval", "question": "Another valid", "answer": "line"}}"#).unwrap();

        let mut eval_stats = HashMap::new();
        let skipped = process_file_for_stats(&file_path, &mut eval_stats).unwrap();

        assert_eq!(skipped, 0); // No valid JSON records were missing eval_key
        let stats = eval_stats.get("test_eval").unwrap();
        assert_eq!(stats.question_count, 2); // Only valid JSON lines counted
        assert_eq!(stats.answer_count, 2);
    }

    #[test]
    fn test_process_file_missing_fields() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_eval.jsonl");

        // Create test file with JSON missing some fields
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, r#"{{"text": "No question or answer"}}"#).unwrap(); // No eval_key, should be skipped
        writeln!(file, r#"{{"eval_key": "test_eval", "question": "Only question"}}"#).unwrap();
        writeln!(file, r#"{{"eval_key": "test_eval", "answer": "Only answer"}}"#).unwrap();
        writeln!(file, r#"{{"eval_key": "test_eval", "passage": "Only passage"}}"#).unwrap();
        writeln!(file, r#"{{"eval_key": "", "question": "Empty eval_key"}}"#).unwrap(); // Empty eval_key, should be skipped

        let mut eval_stats = HashMap::new();
        let skipped = process_file_for_stats(&file_path, &mut eval_stats).unwrap();

        assert_eq!(skipped, 2); // Two records without valid eval_key
        let stats = eval_stats.get("test_eval").unwrap();
        assert_eq!(stats.question_count, 1);
        assert_eq!(stats.answer_count, 1);
        assert_eq!(stats.passage_count, 1);
    }

    #[test]
    fn test_process_compressed_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_eval.jsonl.gz");

        // Create compressed test file
        let file = File::create(&file_path).unwrap();
        let mut encoder = GzEncoder::new(file, Compression::default());
        writeln!(encoder, r#"{{"eval_key": "compressed_eval", "question": "Compressed question", "answer": "Compressed answer"}}"#).unwrap();
        encoder.finish().unwrap();

        let mut eval_stats = HashMap::new();
        let skipped = process_file_for_stats(&file_path, &mut eval_stats).unwrap();

        assert_eq!(skipped, 0);
        let stats = eval_stats.get("compressed_eval").unwrap();
        assert_eq!(stats.question_count, 1);
        assert_eq!(stats.answer_count, 1);
    }

    #[test]
    fn test_eval_key_aggregation() {
        let dir = tempdir().unwrap();

        // Create multiple files with the same eval_key
        let file1 = dir.path().join("file1.jsonl");
        let mut f1 = File::create(&file1).unwrap();
        writeln!(f1, r#"{{"eval_key": "shared_eval", "question": "Q1", "answer": "A1"}}"#).unwrap();
        writeln!(f1, r#"{{"eval_key": "shared_eval", "question": "Q2", "answer": "A2"}}"#).unwrap();

        let file2 = dir.path().join("file2.jsonl.gz");
        let f2 = File::create(&file2).unwrap();
        let mut encoder = GzEncoder::new(f2, Compression::default());
        writeln!(encoder, r#"{{"eval_key": "shared_eval", "question": "Q3", "answer": "A3"}}"#).unwrap();
        writeln!(encoder, r#"{{"eval_key": "different_eval", "question": "Q4", "answer": "A4"}}"#).unwrap();
        encoder.finish().unwrap();

        let mut eval_stats = HashMap::new();
        let skipped1 = process_file_for_stats(&file1, &mut eval_stats).unwrap();
        let skipped2 = process_file_for_stats(&file2, &mut eval_stats).unwrap();

        assert_eq!(skipped1 + skipped2, 0);
        assert_eq!(eval_stats.len(), 2); // Two different eval_keys

        let shared_stats = eval_stats.get("shared_eval").unwrap();
        assert_eq!(shared_stats.question_count, 3); // Aggregated from both files
        assert_eq!(shared_stats.answer_count, 3);

        let different_stats = eval_stats.get("different_eval").unwrap();
        assert_eq!(different_stats.question_count, 1);
        assert_eq!(different_stats.answer_count, 1);
    }

    #[test]
    fn test_collect_eval_stats_nonexistent_dir() {
        let nonexistent = PathBuf::from("/path/that/does/not/exist");
        let result = collect_eval_stats(&nonexistent, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_collect_eval_stats_empty_dir() {
        let dir = tempdir().unwrap();
        let result = collect_eval_stats(dir.path(), false);
        assert!(result.is_ok()); // Should succeed but find no files
    }

    #[test]
    fn test_collect_eval_stats_multiple_files() {
        let dir = tempdir().unwrap();

        // Create multiple test files with eval_keys
        let file1 = dir.path().join("dataset1_test.jsonl");
        let mut f1 = File::create(&file1).unwrap();
        writeln!(f1, r#"{{"eval_key": "dataset1", "question": "Q1", "answer": "A1"}}"#).unwrap();
        writeln!(f1, r#"{{"question": "Q_no_key", "answer": "A_no_key"}}"#).unwrap(); // Will be skipped

        let file2 = dir.path().join("dataset2_train.jsonl");
        let mut f2 = File::create(&file2).unwrap();
        writeln!(f2, r#"{{"eval_key": "dataset2", "question": "Q2", "answer": "A2", "passage": "P2"}}"#).unwrap();

        // This should process both files without errors
        let result = collect_eval_stats(dir.path(), false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_length_calculations() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_lengths.jsonl");

        // Create test file with varying lengths
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, r#"{{"eval_key": "test_eval", "question": "Short", "answer": "S"}}"#).unwrap();
        writeln!(file, r#"{{"eval_key": "test_eval", "question": "Medium length question", "answer": "Med"}}"#).unwrap();
        writeln!(file, r#"{{"eval_key": "test_eval", "question": "This is a very long question that should have the maximum length", "answer": "Long answer here"}}"#).unwrap();

        let mut eval_stats = HashMap::new();
        process_file_for_stats(&file_path, &mut eval_stats).unwrap();

        let stats = eval_stats.get("test_eval").unwrap();

        // Verify min/max calculations work correctly
        let min_q_len = *stats.question_lengths.iter().min().unwrap();
        let max_q_len = *stats.question_lengths.iter().max().unwrap();
        assert_eq!(min_q_len, 5); // "Short"
        assert_eq!(max_q_len, 64); // The long question (64 chars)

        let min_a_len = *stats.answer_lengths.iter().min().unwrap();
        let max_a_len = *stats.answer_lengths.iter().max().unwrap();
        assert_eq!(min_a_len, 1); // "S"
        assert_eq!(max_a_len, 16); // "Long answer here"
    }
}
