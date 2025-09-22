use anyhow::{Error, Result};
use bzip2::read::BzDecoder;
use flate2::read::GzDecoder;
use indicatif::ProgressBar;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use xz2::read::XzDecoder;
use zstd::stream::read::Decoder as ZstdDecoder;

/// Helper function to read compressed files (supporting .gz, .zst, .bz2, and .xz)
pub fn read_compressed_file(path: &PathBuf) -> Result<Vec<u8>, Error> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();

    match path.extension().and_then(|s| s.to_str()) {
        Some("gz") => {
            let mut decoder = GzDecoder::new(file);
            decoder.read_to_end(&mut buffer)?;
        }
        Some("zst") | Some("zstd") => {
            let mut decoder = ZstdDecoder::new(file)?;
            decoder.read_to_end(&mut buffer)?;
        }
        Some("bz2") => {
            let mut decoder = BzDecoder::new(file);
            decoder.read_to_end(&mut buffer)?;
        }
        Some("xz") => {
            let mut decoder = XzDecoder::new(file);
            decoder.read_to_end(&mut buffer)?;
        }
        _ => {
            // No compression, read file directly
            file.read_to_end(&mut buffer)?;
        }
    }

    Ok(buffer)
}

/// Helper function to create progress bar only if not in quiet mode, which prevents
/// unnecessary noise when running integration tests.
pub fn build_pbar_quiet(len: usize, msg: &str) -> ProgressBar {
    if crate::common::is_quiet_mode() {
        ProgressBar::hidden()
    } else {
        mj_io::build_pbar(len, msg)
    }
}

pub fn format_number_with_commas(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();

    for (count, ch) in s.chars().rev().enumerate() {
        if count > 0 && count % 3 == 0 {
            result.push(',');
        }
        result.push(ch);
    }

    result.chars().rev().collect()
}

/// Display timing and traversal statistics
pub fn display_timing_stats(stats: &crate::detect::stats::AggregatedStats) {
    crate::vprintln!("\n=== Traversal Statistics ===");
    crate::vprintln!("Left traversals: {:>15}", format_number_with_commas(stats.total_left_traversals));
    crate::vprintln!("Right traversals: {:>15}", format_number_with_commas(stats.total_right_traversals));
    crate::vprintln!("Total traversals: {:>15}", format_number_with_commas(stats.total_left_traversals + stats.total_right_traversals));
    crate::vprintln!("\n=== Contamination Exclusion Statistics ===");
    crate::vprintln!("Excluded (low IDF threshold): {:>15}", format_number_with_commas(stats.total_excluded_low_idf_threshold));
    crate::vprintln!("Excluded (no answer match): {:>15}", format_number_with_commas(stats.total_excluded_no_answer));
    crate::vprintln!("Excluded (no passage match): {:>15}", format_number_with_commas(stats.total_excluded_no_passage));

    crate::vprintln!("\n=== Core Activity Timing ===");

    if stats.total_question_expansion_calls > 0 {
        let avg_us = stats.total_question_expansion_time_us as f64 / stats.total_question_expansion_calls as f64;
        crate::vprintln!("Question Expansion:");
        crate::vprintln!("  Total time: {:.3}s", stats.total_question_expansion_time_us as f64 / 1_000_000.0);
        crate::vprintln!("  Calls: {:>15}", format_number_with_commas(stats.total_question_expansion_calls));
        crate::vprintln!("  Average: {:.1} μs/call", avg_us);
    }

    if stats.total_passage_cluster_calls > 0 {
        let avg_us = stats.total_passage_cluster_time_us as f64 / stats.total_passage_cluster_calls as f64;
        crate::vprintln!("Passage Cluster Identification:");
        crate::vprintln!("  Total time: {:.3}s", stats.total_passage_cluster_time_us as f64 / 1_000_000.0);
        crate::vprintln!("  Calls: {:>15}", format_number_with_commas(stats.total_passage_cluster_calls));
        crate::vprintln!("  Average: {:.1} μs/call", avg_us);
    }

    if stats.total_answer_cluster_calls > 0 {
        let avg_us = stats.total_answer_cluster_time_us as f64 / stats.total_answer_cluster_calls as f64;
        crate::vprintln!("Answer Cluster Identification:");
        crate::vprintln!("  Total time: {:.3}s", stats.total_answer_cluster_time_us as f64 / 1_000_000.0);
        crate::vprintln!("  Calls: {:>15}", format_number_with_commas(stats.total_answer_cluster_calls));
        crate::vprintln!("  Average: {:.1} μs/call", avg_us);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_format_number_with_commas() {
        assert_eq!(format_number_with_commas(0), "0");
        assert_eq!(format_number_with_commas(1), "1");
        assert_eq!(format_number_with_commas(12), "12");
        assert_eq!(format_number_with_commas(123), "123");
        assert_eq!(format_number_with_commas(1234), "1,234");
        assert_eq!(format_number_with_commas(12345), "12,345");
        assert_eq!(format_number_with_commas(123456), "123,456");
        assert_eq!(format_number_with_commas(1234567), "1,234,567");
        assert_eq!(format_number_with_commas(1234567890), "1,234,567,890");
    }

    #[test]
    fn test_read_uncompressed_file() {
        // Create a temporary file with some content
        let mut temp_file = NamedTempFile::new().unwrap();
        let content = b"Hello, world!\nThis is a test file.";
        temp_file.write_all(content).unwrap();

        let path = temp_file.path().to_path_buf();
        let result = read_compressed_file(&path).unwrap();

        assert_eq!(result, content);
    }

    #[test]
    fn test_read_gzip_file() {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        // Create a temporary gzip file
        let temp_file = NamedTempFile::with_suffix(".gz").unwrap();
        let content = b"Gzipped content\nLine 2";

        let mut encoder = GzEncoder::new(fs::File::create(temp_file.path()).unwrap(), Compression::default());
        encoder.write_all(content).unwrap();
        encoder.finish().unwrap();

        let path = temp_file.path().to_path_buf();
        let result = read_compressed_file(&path).unwrap();

        assert_eq!(result, content);
    }

    #[test]
    fn test_read_zstd_file() {
        use zstd::stream::write::Encoder as ZstdEncoder;

        // Create a temporary zstd file
        let temp_file = NamedTempFile::with_suffix(".zst").unwrap();
        let content = b"Zstd compressed content\nAnother line";

        let mut encoder = ZstdEncoder::new(fs::File::create(temp_file.path()).unwrap(), 3).unwrap();
        encoder.write_all(content).unwrap();
        encoder.finish().unwrap();

        let path = temp_file.path().to_path_buf();
        let result = read_compressed_file(&path).unwrap();

        assert_eq!(result, content);
    }

    #[test]
    fn test_read_bz2_file() {
        use bzip2::write::BzEncoder;
        use bzip2::Compression;

        // Create a temporary bz2 file
        let temp_file = NamedTempFile::with_suffix(".bz2").unwrap();
        let content = b"Bzip2 compressed content\nTest line";

        let mut encoder = BzEncoder::new(fs::File::create(temp_file.path()).unwrap(), Compression::best());
        encoder.write_all(content).unwrap();
        encoder.finish().unwrap();

        let path = temp_file.path().to_path_buf();
        let result = read_compressed_file(&path).unwrap();

        assert_eq!(result, content);
    }

    #[test]
    fn test_read_xz_file() {
        use xz2::write::XzEncoder;

        // Create a temporary xz file
        let temp_file = NamedTempFile::with_suffix(".xz").unwrap();
        let content = b"XZ compressed content\nAnother test line";

        let mut encoder = XzEncoder::new(fs::File::create(temp_file.path()).unwrap(), 6);
        encoder.write_all(content).unwrap();
        encoder.finish().unwrap();

        let path = temp_file.path().to_path_buf();
        let result = read_compressed_file(&path).unwrap();

        assert_eq!(result, content);
    }

    #[test]
    fn test_read_nonexistent_file() {
        let path = PathBuf::from("/tmp/nonexistent_file_that_should_not_exist.txt");
        let result = read_compressed_file(&path);

        assert!(result.is_err());
    }
}
