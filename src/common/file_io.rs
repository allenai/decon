use anyhow::Result;
use std::path::{Path, PathBuf};
use zstd::stream::read::Decoder as ZstdDecoder;

use crate::common::Config;


/* We write contamination reports and cleaned versions of files.
 * The following methods support these operations. */

pub fn generate_report_filename(input_file: &Path, config: &Config, base_input_dir: &Path,) -> Result<String, anyhow::Error> {
    // Calculate relative path from base_input_dir to input_file
    let relative_path = input_file.strip_prefix(base_input_dir).unwrap_or_else(|_| {
        // If strip_prefix fails, just use the filename
        input_file
            .file_name()
            .map(std::path::Path::new)
            .unwrap_or_else(|| std::path::Path::new("unknown"))
    });

    let parent_dir = relative_path.parent();

    // Get base filename without extension
    let base_name = relative_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    // Get the threshold value (mode is always "simple" now)
    let threshold = config.contamination_score_threshold;

    // Construct the result filename
    let result_filename = format!("{}-{}-{:.2}.jsonl", base_name, config.mode, threshold);

    // Combine with parent directory if it exists
    if let Some(parent) = parent_dir {
        if parent.as_os_str().is_empty() {
            Ok(result_filename)
        } else {
            Ok(parent.join(result_filename).to_string_lossy().to_string())
        }
    } else {
        Ok(result_filename)
    }
}

pub fn generate_purified_filename(
    input_file: &Path,
    base_input_dir: &Path,
) -> Result<String, anyhow::Error> {
    // Calculate relative path from base_input_dir to input_file
    let relative_path = input_file.strip_prefix(base_input_dir).unwrap_or_else(|_| {
        // If strip_prefix fails, just use the filename
        input_file
            .file_name()
            .map(std::path::Path::new)
            .unwrap_or_else(|| std::path::Path::new("unknown"))
    });

    // Get parent directory path (if any)
    let parent_dir = relative_path.parent();

    // Get the filename
    let filename = relative_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    // Remove .jsonl extension if present (and any compression extension)
    let base_name = if let Some(pos) = filename.find(".jsonl") {
        &filename[..pos]
    } else {
        // If no .jsonl extension, just use the stem
        relative_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
    };

    // Construct the purified filename
    let purified_filename = format!("{}.jsonl.gz", base_name);

    // Combine with parent directory if it exists
    if let Some(parent) = parent_dir {
        if parent.as_os_str().is_empty() {
            Ok(purified_filename)
        } else {
            Ok(parent.join(purified_filename).to_string_lossy().to_string())
        }
    } else {
        Ok(purified_filename)
    }
}

pub fn write_purified_file(
    input_path: &Path,
    cleaned_output_dir: &Path,
    contaminated_lines: &std::collections::HashSet<usize>,
    base_input_dir: &Path,
) -> Result<PathBuf, anyhow::Error> {
    use std::fs::create_dir_all;

    let purified_filename = generate_purified_filename(input_path, base_input_dir)?;
    let purified_path = cleaned_output_dir.join(&purified_filename);

    // Create parent directories if they don't exist (for preserving directory structure)
    if let Some(parent) = purified_path.parent() {
        create_dir_all(parent)?;
    }

    write_purified_file_internal(input_path, purified_path, contaminated_lines)
}

pub fn write_purified_file_with_job_id(
    input_path: &Path,
    cleaned_output_dir: &Path,
    contaminated_lines: &std::collections::HashSet<usize>,
    job_id: &str,
) -> Result<PathBuf, anyhow::Error> {
    use std::fs::create_dir_all;

    // Ensure output directory exists
    create_dir_all(cleaned_output_dir)?;

    // Use job_id for filename
    let purified_filename = format!("{}.jsonl.gz", job_id);
    let purified_path = cleaned_output_dir.join(&purified_filename);

    write_purified_file_internal(input_path, purified_path, contaminated_lines)
}

// Internal function to write a purified file with contaminated lines removed
fn write_purified_file_internal(
    input_path: &Path,
    output_path: PathBuf,
    contaminated_lines: &std::collections::HashSet<usize>,
) -> Result<PathBuf, anyhow::Error> {
    use flate2::read::GzDecoder;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::fs::File;
    use std::io::{BufRead, BufReader, BufWriter, Write};

    // Open input file with streaming reader
    let file = File::open(input_path)?;
    let mut reader: Box<dyn BufRead> = match input_path.extension().and_then(|s| s.to_str()) {
        Some("gz") => Box::new(BufReader::new(GzDecoder::new(file))),
        Some("zst") | Some("zstd") => Box::new(BufReader::new(ZstdDecoder::new(file)?)),
        _ => Box::new(BufReader::new(file)),
    };

    // Create output file with gzip compression
    let output_file = File::create(&output_path)?;
    let gz_encoder = GzEncoder::new(output_file, Compression::default());
    let mut writer = BufWriter::new(gz_encoder);

    // Work with raw bytes to preserve data exactly as-is
    let mut line_num = 0;
    let mut line_buffer = Vec::new();

    loop {
        line_buffer.clear();
        let bytes_read = reader.read_until(b'\n', &mut line_buffer)?;
        if bytes_read == 0 {
            break;
        }

        if !contaminated_lines.contains(&line_num) {
            // Write bytes exactly as they are, no UTF-8 validation
            writer.write_all(&line_buffer)?;
        }
        line_num += 1;
    }

    writer.flush()?;

    Ok(output_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Config;
    use std::collections::HashSet;
    use std::fs;
    use std::io::Write;
    use tempfile::tempdir;

    fn create_test_config(mode: &str, contamination_threshold: f32) -> Config {
        use crate::detect::scoring::calculate_minimum_question_idf_threshold;
        let contamination_score_threshold = contamination_threshold;
        Config {
            mode: mode.to_string(),
            ngram_size: 5,
            tokenizer_str: "cl100k".to_string(),
            hash_seed: 0,
            content_key: "text".to_string(),
            training_dir: PathBuf::from("/tmp/input"),
            evals_dir: PathBuf::from("/tmp/ref"),
            report_output_dir: PathBuf::from("/tmp/report"),
            cleaned_output_dir: None,
            exact_override: false,
            sample_every_m_tokens: 10,
            question_max_consecutive_misses: 11,
            ngram_bucket_lru_cache: 10000,
            punctuation_chars: String::new(),
            worker_threads: 4,
            window_size_increment: None,
            num_windows: None,
            window_step_size: None,
            short_answer_window_length: 50,
            min_long_answer_window: 100,
            short_answer_token_threshold: 3,
            answer_ngram_size: 3,
            verbose: false,
            min_passage_distance: 100,
            passage_max_consecutive_misses: 2,
            passage_ngram_size: 4,
            contamination_score_threshold,
            minimum_question_idf_threshold: calculate_minimum_question_idf_threshold(contamination_score_threshold),
            purify: false,
            eval_dedup: true,
            eval_min_token_length: 20,
            eval_min_unique_word_count: 4,
            perfect_match_decay_start: Some(20),
            perfect_match_decay_end: Some(50),
            index_passages: true,
            index_answers: true,
        }
    }

    #[test]
    fn test_generate_report_filename_with_simple_mode() {
        let config = create_test_config("simple", 0.75);

        let base_dir = PathBuf::from("/data/input");
        let input_file = PathBuf::from("/data/input/subdir/test.jsonl");

        let result = generate_report_filename(&input_file, &config, &base_dir).unwrap();
        assert_eq!(result, "subdir/test-simple-0.75.jsonl");
    }

    #[test]
    fn test_generate_report_filename_with_other_mode() {
        let config = create_test_config("minhash", 0.75);

        let base_dir = PathBuf::from("/data/input");
        let input_file = PathBuf::from("/data/input/test.jsonl");

        let result = generate_report_filename(&input_file, &config, &base_dir).unwrap();
        assert_eq!(result, "test-minhash-0.75.jsonl");
    }

    #[test]
    fn test_generate_report_filename_strip_prefix_fails() {
        let config = create_test_config("simple", 0.5);

        let base_dir = PathBuf::from("/different/path");
        let input_file = PathBuf::from("/data/input/test.jsonl");

        let result = generate_report_filename(&input_file, &config, &base_dir).unwrap();
        assert_eq!(result, "test-simple-0.50.jsonl");
    }

    #[test]
    fn test_generate_purified_filename_with_jsonl_gz() {
        let base_dir = PathBuf::from("/data/input");
        let input_file = PathBuf::from("/data/input/subdir/test.jsonl.gz");

        let result = generate_purified_filename(&input_file, &base_dir).unwrap();
        assert_eq!(result, "subdir/test.jsonl.gz");
    }

    #[test]
    fn test_generate_purified_filename_with_jsonl() {
        let base_dir = PathBuf::from("/data/input");
        let input_file = PathBuf::from("/data/input/test.jsonl");

        let result = generate_purified_filename(&input_file, &base_dir).unwrap();
        assert_eq!(result, "test.jsonl.gz");
    }

    #[test]
    fn test_generate_purified_filename_without_jsonl() {
        let base_dir = PathBuf::from("/data/input");
        let input_file = PathBuf::from("/data/input/test.txt");

        let result = generate_purified_filename(&input_file, &base_dir).unwrap();
        assert_eq!(result, "test.jsonl.gz");
    }

    #[test]
    fn test_write_purified_file_basic() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.jsonl");
        let output_dir = dir.path().join("output");
        let base_dir = dir.path().to_path_buf();

        // Create input file with some lines
        let mut file = fs::File::create(&input_path).unwrap();
        writeln!(file, "{{\"text\": \"line 0\"}}").unwrap();
        writeln!(file, "{{\"text\": \"line 1\"}}").unwrap();
        writeln!(file, "{{\"text\": \"line 2\"}}").unwrap();

        // Mark line 1 as contaminated
        let mut contaminated = HashSet::new();
        contaminated.insert(1);

        let result = write_purified_file(&input_path, &output_dir, &contaminated, &base_dir).unwrap();

        // Check that output file exists
        assert!(result.exists());
        assert!(result.to_string_lossy().ends_with(".jsonl.gz"));

        // Read and decompress to verify content
        use flate2::read::GzDecoder;
        use std::io::{BufRead, BufReader};

        let file = fs::File::open(&result).unwrap();
        let decoder = GzDecoder::new(file);
        let reader = BufReader::new(decoder);
        let lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();

        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "{\"text\": \"line 0\"}");
        assert_eq!(lines[1], "{\"text\": \"line 2\"}");
    }

    #[test]
    fn test_write_purified_file_with_job_id_basic() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.jsonl");
        let output_dir = dir.path().join("output");

        // Create input file
        let mut file = fs::File::create(&input_path).unwrap();
        writeln!(file, "{{\"text\": \"line 0\"}}").unwrap();
        writeln!(file, "{{\"text\": \"line 1\"}}").unwrap();

        let contaminated = HashSet::new(); // No contamination

        let result = write_purified_file_with_job_id(&input_path, &output_dir, &contaminated, "job123").unwrap();

        // Check filename uses job_id
        assert_eq!(result.file_name().unwrap().to_str().unwrap(), "job123.jsonl.gz");
        assert!(result.exists());
    }

    #[test]
    fn test_write_purified_file_creates_directories() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.jsonl");
        let output_dir = dir.path().join("deep/nested/output");
        let base_dir = dir.path().to_path_buf();

        // Create input file
        fs::File::create(&input_path).unwrap();

        let contaminated = HashSet::new();

        let result = write_purified_file(&input_path, &output_dir, &contaminated, &base_dir).unwrap();

        // Check that nested directories were created
        assert!(output_dir.exists());
        assert!(result.exists());
    }
}
