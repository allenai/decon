use clap::Args;
use std::path::PathBuf;

#[derive(Args, Debug)]
#[command(about = "Compare contamination results from two detection runs")]
pub struct CompareArgs {
    #[arg(help = "First directory containing contamination results")]
    pub dir1: PathBuf,

    #[arg(help = "Second directory containing contamination results")]
    pub dir2: PathBuf,

    #[arg(
        long,
        help = "Show comparative statistics between the two runs"
    )]
    pub stats: bool,

    #[arg(
        long,
        help = "Show contaminations present in both directories"
    )]
    pub common: bool,

    #[arg(
        long,
        help = "Show only fingerprints that are in the first directory but not in the second"
    )]
    pub only_in_first: bool,

    #[arg(
        long,
        help = "Show only fingerprints that are in the second directory but not in the first"
    )]
    pub only_in_second: bool,

    #[arg(
        long,
        help = "Minimum contamination score to include in results"
    )]
    pub min_score: Option<f32>,

    #[arg(
        long,
        help = "Filter by evaluation dataset name, e.g. mmlu"
    )]
    pub eval: Option<String>,

    #[arg(
        long,
        short = 'v',
        help = "Show verbose output with all scores and metrics"
    )]
    pub verbose: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_args_creation() {
        let args = CompareArgs {
            dir1: PathBuf::from("/path/to/dir1"),
            dir2: PathBuf::from("/path/to/dir2"),
            stats: false,
            common: false,
            only_in_first: false,
            only_in_second: false,
            min_score: None,
            eval: None,
            verbose: false,
        };

        assert_eq!(args.dir1, PathBuf::from("/path/to/dir1"));
        assert_eq!(args.dir2, PathBuf::from("/path/to/dir2"));
        assert!(!args.stats);
        assert!(!args.common);
        assert!(!args.only_in_first);
        assert!(!args.only_in_second);
        assert!(args.min_score.is_none());
        assert!(args.eval.is_none());
        assert!(!args.verbose);
    }

    #[test]
    fn test_compare_args_with_options() {
        let args = CompareArgs {
            dir1: PathBuf::from("/path/to/dir1"),
            dir2: PathBuf::from("/path/to/dir2"),
            stats: true,
            common: true,
            only_in_first: false,
            only_in_second: false,
            min_score: Some(0.75),
            eval: Some("mmlu".to_string()),
            verbose: true,
        };

        assert!(args.stats);
        assert!(args.common);
        assert_eq!(args.min_score, Some(0.75));
        assert_eq!(args.eval, Some("mmlu".to_string()));
        assert!(args.verbose);
    }

    #[test]
    fn test_compare_args_all_flags_enabled() {
        let args = CompareArgs {
            dir1: PathBuf::from("/path/to/dir1"),
            dir2: PathBuf::from("/path/to/dir2"),
            stats: true,
            common: true,
            only_in_first: true,
            only_in_second: true,
            min_score: Some(0.5),
            eval: Some("gsm8k".to_string()),
            verbose: true,
        };

        assert!(args.stats);
        assert!(args.common);
        assert!(args.only_in_first);
        assert!(args.only_in_second);
        assert_eq!(args.min_score, Some(0.5));
        assert_eq!(args.eval, Some("gsm8k".to_string()));
        assert!(args.verbose);
    }

    #[test]
    fn test_compare_args_paths() {
        let args = CompareArgs {
            dir1: PathBuf::from("./relative/path1"),
            dir2: PathBuf::from("../relative/path2"),
            stats: false,
            common: false,
            only_in_first: false,
            only_in_second: false,
            min_score: None,
            eval: None,
            verbose: false,
        };

        assert_eq!(args.dir1.to_str().unwrap(), "./relative/path1");
        assert_eq!(args.dir2.to_str().unwrap(), "../relative/path2");
    }

    #[test]
    fn test_compare_args_absolute_paths() {
        let args = CompareArgs {
            dir1: PathBuf::from("/absolute/path/dir1"),
            dir2: PathBuf::from("/absolute/path/dir2"),
            stats: false,
            common: false,
            only_in_first: false,
            only_in_second: false,
            min_score: None,
            eval: None,
            verbose: false,
        };

        assert!(args.dir1.is_absolute());
        assert!(args.dir2.is_absolute());
    }

    #[test]
    fn test_compare_args_min_score_boundary_values() {
        let args_zero = CompareArgs {
            dir1: PathBuf::from("/path/to/dir1"),
            dir2: PathBuf::from("/path/to/dir2"),
            stats: false,
            common: false,
            only_in_first: false,
            only_in_second: false,
            min_score: Some(0.0),
            eval: None,
            verbose: false,
        };

        let args_one = CompareArgs {
            dir1: PathBuf::from("/path/to/dir1"),
            dir2: PathBuf::from("/path/to/dir2"),
            stats: false,
            common: false,
            only_in_first: false,
            only_in_second: false,
            min_score: Some(1.0),
            eval: None,
            verbose: false,
        };

        assert_eq!(args_zero.min_score, Some(0.0));
        assert_eq!(args_one.min_score, Some(1.0));
    }
}
