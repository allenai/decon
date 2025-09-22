use std::collections::HashMap;

/// Statistics about hot buckets (n-grams that map to many documents)
/// Hot buckets get special treatment to optimize the hot path of detecting and starting
/// cluster expansion. Because they are so common, they are poor candidates to start
/// clusters on, rather we increment training document index by 1 until we find a more
/// informative n-gram. These statistics help to study the phenomenon and are used
/// when running in verbose mode with timing and counting enabled.
#[derive(Clone)]
pub struct HotBucketStats {
    pub total_ngrams: usize,
    pub bucket_histogram: Vec<(usize, usize, String, usize)>, // Histogram of bucket sizes: (min, max, label) -> count
    pub mean_bucket_size: f64, // (documents per n-gram)
    pub median_bucket_size: usize,
    pub max_bucket_size: usize,
    pub hot_ngram_examples: Vec<(String, usize)>, // Examples of hot n-grams with their text and document counts
}

impl HotBucketStats {
    /// Analyze the n-gram to document mapping and create statistics
    pub fn analyze(
        ngram_id_to_docs: &HashMap<u32, std::collections::HashSet<u32>>,
        hot_ngram_examples: Vec<(String, usize)>,
    ) -> Self {
        let total_ngrams = ngram_id_to_docs.len();

        // Collect all bucket sizes
        let mut bucket_sizes: Vec<usize> = ngram_id_to_docs
            .values()
            .map(|docs| docs.len())
            .collect();

        // Sort for median calculation
        bucket_sizes.sort_unstable();

        // Calculate statistics
        let sum: usize = bucket_sizes.iter().sum();
        let mean_bucket_size = if !bucket_sizes.is_empty() {
            sum as f64 / bucket_sizes.len() as f64
        } else {
            0.0
        };

        let median_bucket_size = if !bucket_sizes.is_empty() {
            bucket_sizes[bucket_sizes.len() / 2]
        } else {
            0
        };

        let max_bucket_size = bucket_sizes.iter().max().copied().unwrap_or(0);

        // Create histogram buckets
        let buckets: Vec<(usize, usize, &str)> = vec![
            (1, 10, "1-10 docs"),
            (11, 50, "11-50 docs"),
            (51, 100, "51-100 docs"),
            (101, 500, "101-500 docs"),
            (501, usize::MAX, "500+ docs"),
        ];

        // Count documents in each bucket
        let mut bucket_histogram = Vec::new();
        for (min, max, label) in buckets {
            let count = bucket_sizes
                .iter()
                .filter(|&&size| size >= min && size <= max.min(usize::MAX - 1))
                .count();
            bucket_histogram.push((min, max, label.to_string(), count));
        }

        Self {
            total_ngrams,
            bucket_histogram,
            mean_bucket_size,
            median_bucket_size,
            max_bucket_size,
            hot_ngram_examples,
        }
    }

    /// Display the hot bucket statistics as a histogram
    pub fn display(&self) {
        println!("=== HOT BUCKET ANALYSIS ===");
        println!("Total unique n-grams: {}", self.total_ngrams);
        println!();

        if self.total_ngrams == 0 {
            println!("No n-grams in index.");
            return;
        }

        println!("Ngram -> Document Count Distribution:");

        // Find the maximum count for scaling the histogram
        let max_count = self.bucket_histogram
            .iter()
            .map(|(_, _, _, count)| *count)
            .max()
            .unwrap_or(0);

        // Display histogram
        let bar_width = 40;
        for (_, _, label, count) in &self.bucket_histogram {
            let percentage = (*count as f64 / self.total_ngrams as f64) * 100.0;

            // Calculate bar length proportional to count
            let bar_length = if max_count > 0 {
                ((*count as f64 / max_count as f64) * bar_width as f64) as usize
            } else {
                0
            };

            // Create the bar using Unicode block characters
            let bar = "█".repeat(bar_length);
            let empty = " ".repeat(bar_width - bar_length);

            // Format the output with aligned columns
            println!("  {:<15} {:>8} ({:>5.1}%) │{}{}│",
                label, count, percentage, bar, empty);
        }

        println!();
        println!("Statistics:");
        println!("  Mean bucket size: {:.1} documents", self.mean_bucket_size);
        println!("  Median: {} documents", self.median_bucket_size);
        println!("  Max: {} documents", self.max_bucket_size);


        println!();
    }
}
