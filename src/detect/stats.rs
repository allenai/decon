use std::cell::RefCell;
use std::sync::Arc;
use std::time::Instant;
use thread_local::ThreadLocal;

#[derive(Debug, Clone, Default)]
pub struct ThreadLocalStats {
    pub files_processed: usize,
    pub lines_processed: usize,
    pub contaminations_found: usize,
    // Timing accumulation (microseconds)
    pub question_expansion_time_us: u64,
    pub answer_cluster_time_us: u64,
    pub passage_cluster_time_us: u64,
    // Operation counts
    pub question_expansion_calls: usize,
    pub answer_cluster_calls: usize,
    pub passage_cluster_calls: usize,
    // Traversal statistics
    pub left_traversals: usize,
    pub right_traversals: usize,
    // Exclusion statistics
    pub excluded_no_answer: usize,
    pub excluded_no_passage: usize,
    pub excluded_low_idf_threshold: usize,
}

#[derive(Debug, Clone, Default)]
pub struct AggregatedStats {
    pub total_files_processed: usize,
    pub total_lines_processed: usize,
    pub total_contaminations_found: usize,
    pub total_question_expansion_time_us: u64,
    pub total_answer_cluster_time_us: u64,
    pub total_passage_cluster_time_us: u64,
    pub total_question_expansion_calls: usize,
    pub total_answer_cluster_calls: usize,
    pub total_passage_cluster_calls: usize,
    pub total_left_traversals: usize,
    pub total_right_traversals: usize,
    pub total_excluded_no_answer: usize,
    pub total_excluded_no_passage: usize,
    pub total_excluded_low_idf_threshold: usize,
}

pub struct StatsContainer {
    stats: Arc<ThreadLocal<RefCell<ThreadLocalStats>>>,
    start_time: Instant,
}

impl Default for StatsContainer {
    fn default() -> Self {
        Self::new()
    }
}

impl StatsContainer {
    pub fn new() -> Self {
        Self {
            stats: Arc::new(ThreadLocal::new()),
            start_time: Instant::now(),
        }
    }

    fn get_local_stats(&self) -> &RefCell<ThreadLocalStats> {
        self.stats.get_or(|| RefCell::new(ThreadLocalStats::default()))
    }

    pub fn increment_files_processed(&self) {
        let cell = self.get_local_stats();
        let mut stats = cell.borrow_mut();
        stats.files_processed += 1;
    }

    pub fn add_lines_processed(&self, lines: usize) {
        let cell = self.get_local_stats();
        let mut stats = cell.borrow_mut();
        stats.lines_processed += lines;
    }

    pub fn add_contaminations_found(&self, count: usize) {
        let cell = self.get_local_stats();
        let mut stats = cell.borrow_mut();
        stats.contaminations_found += count;
    }

    pub fn add_question_expansion_time(&self, us: u64) {
        let cell = self.get_local_stats();
        let mut stats = cell.borrow_mut();
        stats.question_expansion_time_us += us;
        stats.question_expansion_calls += 1;
    }

    pub fn add_answer_cluster_time(&self, us: u64) {
        let cell = self.get_local_stats();
        let mut stats = cell.borrow_mut();
        stats.answer_cluster_time_us += us;
        stats.answer_cluster_calls += 1;
    }

    pub fn add_passage_cluster_time(&self, us: u64) {
        let cell = self.get_local_stats();
        let mut stats = cell.borrow_mut();
        stats.passage_cluster_time_us += us;
        stats.passage_cluster_calls += 1;
    }

    pub fn increment_left_traversals(&self) {
        let cell = self.get_local_stats();
        let mut stats = cell.borrow_mut();
        stats.left_traversals += 1;
    }

    pub fn increment_right_traversals(&self) {
        let cell = self.get_local_stats();
        let mut stats = cell.borrow_mut();
        stats.right_traversals += 1;
    }

    pub fn increment_excluded_no_answer(&self) {
        let cell = self.get_local_stats();
        let mut stats = cell.borrow_mut();
        stats.excluded_no_answer += 1;
    }

    pub fn increment_excluded_no_passage(&self) {
        let cell = self.get_local_stats();
        let mut stats = cell.borrow_mut();
        stats.excluded_no_passage += 1;
    }

    pub fn increment_excluded_low_idf_threshold(&self) {
        let cell = self.get_local_stats();
        let mut stats = cell.borrow_mut();
        stats.excluded_low_idf_threshold += 1;
    }

    pub fn aggregate(self) -> AggregatedStats {
        // Try to unwrap the Arc to get ownership of the ThreadLocal
        let stats = match Arc::try_unwrap(self.stats) {
            Ok(tl) => tl,
            Err(_arc) => {
                // This shouldn't happen in normal use since we consume self
                eprintln!("Warning: StatsContainer still has multiple references during aggregation");
                // Create a new empty ThreadLocal as fallback
                // In production, this case should be avoided by proper lifecycle management
                return AggregatedStats::default();
            }
        };

        let thread_stats: Vec<ThreadLocalStats> = stats
            .into_iter()
            .map(|refcell| refcell.into_inner())
            .filter(|s| s.files_processed > 0 || s.lines_processed > 0)
            .collect();

        let mut aggregated = AggregatedStats::default();

        for stats in &thread_stats {
            aggregated.total_files_processed += stats.files_processed;
            aggregated.total_lines_processed += stats.lines_processed;
            aggregated.total_contaminations_found += stats.contaminations_found;
            aggregated.total_question_expansion_time_us += stats.question_expansion_time_us;
            aggregated.total_answer_cluster_time_us += stats.answer_cluster_time_us;
            aggregated.total_passage_cluster_time_us += stats.passage_cluster_time_us;
            aggregated.total_question_expansion_calls += stats.question_expansion_calls;
            aggregated.total_answer_cluster_calls += stats.answer_cluster_calls;
            aggregated.total_passage_cluster_calls += stats.passage_cluster_calls;
            aggregated.total_left_traversals += stats.left_traversals;
            aggregated.total_right_traversals += stats.right_traversals;
            aggregated.total_excluded_no_answer += stats.excluded_no_answer;
            aggregated.total_excluded_no_passage += stats.excluded_no_passage;
            aggregated.total_excluded_low_idf_threshold += stats.excluded_low_idf_threshold;
        }

        aggregated
    }
}

impl Clone for StatsContainer {
    fn clone(&self) -> Self {
        Self {
            stats: Arc::clone(&self.stats),
            start_time: self.start_time,
        }
    }
}