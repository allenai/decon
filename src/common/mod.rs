pub mod detection_config;
pub mod file_io;
pub mod json_util;
pub mod text;
pub mod tokenizer;

// Re-export commonly used items for convenience
pub use detection_config::{Config, read_config, create_default_config, validate_config, generate_temp_dir};
pub use file_io::{
    generate_report_filename, write_purified_file, write_purified_file_with_job_id,
};
pub use json_util::get_nested_json_val;
pub use text::{clean_text, default_punctuation_chars};
pub use tokenizer::OmniTokenizer;

/// Supported file extensions for data files (JSON/JSONL with various compression formats)
pub const SUPPORTED_DATA_EXTENSIONS: &[&str] = &[
    ".json", ".json.gz", ".json.zst", ".json.zstd", ".json.bz2", ".json.xz",
    ".jsonl", ".jsonl.gz", ".jsonl.zst", ".jsonl.zstd", ".jsonl.bz2", ".jsonl.xz",
];

/// Extensions passed to expand_dirs for finding data files
/// Note: expand_dirs matches on suffix, so we use shorter forms that will match
/// compressed variants (e.g., ".jsonl" matches ".jsonl.gz", ".jsonl.zst", etc.)
pub const EXPAND_DIRS_EXTENSIONS: &[&str] = &[".json", ".jsonl", ".gz", ".zst", ".zstd", ".bz2", ".xz"];

/// Check if quiet mode is enabled (suppress output)
pub fn is_quiet_mode() -> bool {
    std::env::var("DECON_QUIET").is_ok() || std::env::var("DECON_TEST").is_ok()
}

/// Macro for conditional printing - only prints if not in quiet mode
#[macro_export]
macro_rules! vprintln {
    ($($arg:tt)*) => {
        if !$crate::common::is_quiet_mode() {
            println!($($arg)*);
        }
    }
}