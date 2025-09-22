use clap::Args;
use std::path::PathBuf;

/// Common arguments shared between Detect and Server commands
#[derive(Args, Debug)]
pub struct CommonDetectionArgs {
    // Input and Output
    #[arg(
        long,
        help = "Directory containing training data in jsonl format [required]",
        display_order = 1,
        help_heading = "Input and Output"
    )]
    pub training_dir: Option<PathBuf>,

    #[arg(
        long,
        help = "JSON field containing text content in training documents [default: text]",
        display_order = 2,
        help_heading = "Input and Output"
    )]
    pub content_key: Option<String>,

    #[arg(
        long,
        help = "Directory containing evaluation datasets in decon expected format [default: bundled-evals]",
        display_order = 3,
        help_heading = "Input and Output"
    )]
    pub evals_dir: Option<PathBuf>,

    #[arg(
        long,
        help = "Output directory for contamination reports [default: auto-generated in /tmp]",
        display_order = 4,
        help_heading = "Input and Output"
    )]
    pub report_output_dir: Option<PathBuf>,

    #[arg(
        long,
        help = "Output directory for cleaned files when --purify option is set [default: auto-generated in /tmp when --purify is used]",
        display_order = 5,
        help_heading = "Input and Output"
    )]
    pub cleaned_output_dir: Option<PathBuf>,

    #[arg(
        long,
        help = "Produce cleaned files with contaminated lines removed [default: false]",
        display_order = 6,
        help_heading = "Input and Output"
    )]
    pub purify: bool,

    #[arg(
        long,
        short = 'c',
        help = "Path to YAML configuration file (optional, will use defaults if not provided)",
        display_order = 7
    )]
    pub config: Option<PathBuf>,

    // Matching and Scoring
    #[arg(
        long,
        help = "Tokenizer type: r50k, p50k, p50k_edit, cl100k, o200k, or uniseg [default: cl100k]",
        display_order = 8,
        help_heading = "Matching and Scoring"
    )]
    pub tokenizer: Option<String>,

    #[arg(
        long,
        help = "N-gram size for SIMPLE mode [default: 5]",
        display_order = 9,
        help_heading = "Matching and Scoring"
    )]
    pub ngram_size: Option<usize>,

    #[arg(
        long,
        help = "Sample every M tokens for SIMPLE mode [default: ngram_size + 1]",
        display_order = 10,
        help_heading = "Matching and Scoring"
    )]
    pub sample_every_m_tokens: Option<usize>,

    #[arg(
        long,
        help = "Max consecutive misses before stopping question cluster expansion [default: 11]",
        display_order = 11,
        help_heading = "Matching and Scoring"
    )]
    pub question_max_consecutive_misses: Option<usize>,

    #[arg(
        long,
        help = "Combined contamination score threshold for final decision [default: 0.8]",
        display_order = 14,
        help_heading = "Matching and Scoring"
    )]
    pub contamination_score_threshold: Option<f32>,


    #[arg(
        long,
        help = "Token threshold for short answer exact matching (answers with <= this many tokens use exact matching) [default: 3]",
        display_order = 15,
        help_heading = "Matching and Scoring"
    )]
    pub short_answer_token_threshold: Option<usize>,

    #[arg(
        long,
        help = "Window length for short answer exact matching [default: 50]",
        display_order = 16,
        help_heading = "Matching and Scoring"
    )]
    pub short_answer_window_length: Option<usize>,

    #[arg(
        long,
        help = "Minimum window size for long answer n-gram matching [default: 100]",
        display_order = 17,
        help_heading = "Matching and Scoring"
    )]
    pub min_long_answer_window: Option<usize>,

    #[arg(
        long,
        help = "N-gram size for answer matching [default: 3]",
        display_order = 18,
        help_heading = "Matching and Scoring"
    )]
    pub answer_ngram_size: Option<usize>,

    #[arg(
        long,
        help = "Max consecutive misses before stopping passage expansion [default: 2]",
        display_order = 19,
        help_heading = "Matching and Scoring"
    )]
    pub passage_max_consecutive_misses: Option<usize>,

    #[arg(
        long,
        help = "N-gram size for passage matching [default: 4]",
        display_order = 20,
        help_heading = "Matching and Scoring"
    )]
    pub passage_ngram_size: Option<usize>,

    #[arg(
        long,
        help = "Number of worker threads [default: number of CPU cores]",
        display_order = 21,
        help_heading = "Performance"
    )]
    pub worker_threads: Option<usize>,

    // Reference preprocessing options
    #[arg(
        long,
        help = "Enable exact deduplication of reference entries [default: true]",
        display_order = 22,
        help_heading = "Reference Preprocessing"
    )]
    pub eval_dedup: bool,

    #[arg(
        long,
        help = "Index passages during reference index building [default: true]",
        display_order = 23,
        help_heading = "Reference Preprocessing"
    )]
    pub index_passages: Option<bool>,

    #[arg(
        long,
        help = "Index answers during reference index building [default: true]",
        display_order = 24,
        help_heading = "Reference Preprocessing"
    )]
    pub index_answers: Option<bool>,

    #[arg(
        long,
        help = "Minimum token count for reference entries (0 = disabled) [default: 20]",
        display_order = 25,
        help_heading = "Reference Preprocessing"
    )]
    pub eval_min_token_length: Option<usize>,

    #[arg(
        long,
        help = "Minimum unique word count for reference entries (0 = disabled) [default: 4]",
        display_order = 26,
        help_heading = "Reference Preprocessing"
    )]
    pub eval_min_unique_word_count: Option<usize>,


    #[arg(
        long,
        help = "Cumulative token length where perfect match (score=1.0) requirement starts [default: 20]",
        display_order = 27,
        help_heading = "Matching and Scoring"
    )]
    pub perfect_match_decay_start: Option<usize>,

    #[arg(
        long,
        help = "Cumulative token length where interpolation ends and normal threshold applies [default: 50]",
        display_order = 28,
        help_heading = "Matching and Scoring"
    )]
    pub perfect_match_decay_end: Option<usize>,

    #[arg(
        long,
        short = 'v',
        help = "Enable verbose output (collects and shows timing and counters)",
        display_order = 29,
        help_heading = "Display Options"
    )]
    pub verbose: bool,
}

/// Apply common detection overrides from command-line args to config
pub fn apply_common_detection_overrides(loaded_config: &mut crate::common::Config, args: &CommonDetectionArgs) {
    // Apply command-line overrides
    if let Some(ref ck) = args.content_key {
        loaded_config.content_key = ck.clone();
    }
    if let Some(ref td) = args.training_dir {
        loaded_config.training_dir = td.clone();
    }
    // Evaluation directory resolution with precedence:
    // 1. User-provided --evals-dir flag
    // 2. ~/.local/share/decon/references if it exists and has files
    // 3. Default bundled evals
    if let Some(ref ed) = args.evals_dir {
        // User explicitly provided a directory
        loaded_config.evals_dir = ed.clone();
        eprintln!("Using evaluation directory from --evals-dir: {}\n", ed.display());
    } else {
        // Check for user-downloaded references
        let user_ref_dir = dirs::data_local_dir()
            .map(|d| d.join("decon").join("references"))
            .unwrap_or_else(|| std::path::PathBuf::from("~/.local/share/decon/references"));

        if user_ref_dir.exists() && user_ref_dir.is_dir() {
            // Check if it contains any .jsonl files
            let has_files = std::fs::read_dir(&user_ref_dir)
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
                loaded_config.evals_dir = user_ref_dir.clone();
                eprintln!("Using user-downloaded evaluation directory: {}\n", user_ref_dir.display());
            } else {
                eprintln!("Using default evaluation directory: {}\n", loaded_config.evals_dir.display());
            }
        } else {
            eprintln!("Using default evaluation directory: {}\n", loaded_config.evals_dir.display());
        }
    }
    if let Some(ref rod) = args.report_output_dir {
        loaded_config.report_output_dir = rod.clone();
    }
    if let Some(ref cod) = args.cleaned_output_dir {
        loaded_config.cleaned_output_dir = Some(cod.clone());
    }
    loaded_config.purify = args.purify;

    // Warn if cleaned_output_dir is set but purify is not enabled
    if args.cleaned_output_dir.is_some() && !args.purify {
        eprintln!("\n⚠️  WARNING: --cleaned-output-dir is set but --purify is not enabled!");
        eprintln!("    Cleaned datasets will NOT be generated without --purify.");
        eprintln!("    Add --purify to enable decontaminated dataset generation.\n");
    }

    if let Some(ref t) = args.tokenizer {
        loaded_config.tokenizer_str = t.clone();
    }

    // SIMPLE mode overrides
    if let Some(ns) = args.ngram_size {
        loaded_config.ngram_size = ns;
    }
    if let Some(semt) = args.sample_every_m_tokens {
        loaded_config.sample_every_m_tokens = semt;
    }
    // Ensure sample_every_m_tokens is at least 1
    if loaded_config.sample_every_m_tokens < 1 {
        eprintln!("Warning: sample_every_m_tokens was {}, adjusting to 1 (minimum value)",
                  loaded_config.sample_every_m_tokens);
        loaded_config.sample_every_m_tokens = 1;
    }

    // Warn if sample_every_m_tokens is very large
    if loaded_config.sample_every_m_tokens > 100 {
        eprintln!("Warning: sample_every_m_tokens is set to {}, which is quite large. \
                   This may cause contamination to be missed. Consider using a smaller value.",
                  loaded_config.sample_every_m_tokens);
    }
    if let Some(qmcm) = args.question_max_consecutive_misses {
        loaded_config.question_max_consecutive_misses = qmcm;
    }
    if let Some(cst) = args.contamination_score_threshold {
        loaded_config.contamination_score_threshold = cst;
    }
    if let Some(satt) = args.short_answer_token_threshold {
        loaded_config.short_answer_token_threshold = satt;
    }
    if let Some(sawl) = args.short_answer_window_length {
        loaded_config.short_answer_window_length = sawl;
    }
    if let Some(mlaw) = args.min_long_answer_window {
        loaded_config.min_long_answer_window = mlaw;
    }
    if let Some(ans) = args.answer_ngram_size {
        loaded_config.answer_ngram_size = ans;
    }
    if let Some(pmcm) = args.passage_max_consecutive_misses {
        loaded_config.passage_max_consecutive_misses = pmcm;
    }
    if let Some(pns) = args.passage_ngram_size {
        loaded_config.passage_ngram_size = pns;
    }
    if let Some(wt) = args.worker_threads {
        loaded_config.worker_threads = wt;
    }

    // Reference preprocessing overrides
    if args.eval_dedup {
        loaded_config.eval_dedup = true;
    }
    if let Some(ip) = args.index_passages {
        loaded_config.index_passages = ip;
    }
    if let Some(ia) = args.index_answers {
        loaded_config.index_answers = ia;
    }
    if let Some(eml) = args.eval_min_token_length {
        loaded_config.eval_min_token_length = eml;
    }
    if let Some(emuwc) = args.eval_min_unique_word_count {
        loaded_config.eval_min_unique_word_count = emuwc;
    }
    if let Some(pmds) = args.perfect_match_decay_start {
        loaded_config.perfect_match_decay_start = Some(pmds);
    }
    if let Some(pmde) = args.perfect_match_decay_end {
        loaded_config.perfect_match_decay_end = Some(pmde);
    }
    if args.verbose {
        loaded_config.verbose = true;
    }

    // Compute the minimum question IDF threshold based on contamination_score_threshold
    use crate::detect::scoring::calculate_minimum_question_idf_threshold;
    loaded_config.minimum_question_idf_threshold =
        calculate_minimum_question_idf_threshold(loaded_config.contamination_score_threshold);
}
