use std::path::PathBuf;

fn get_purified_filename(input_file: &PathBuf) -> String {
    // Get the full filename
    let filename = input_file
        .file_name()
        .and_then( < /dev/null | s| s.to_str())
        .unwrap_or("unknown");
    
    // Remove .jsonl extension if present (and any compression extension)
    let base_name = if let Some(pos) = filename.find(".jsonl") {
        &filename[..pos]
    } else {
        // If no .jsonl extension, just use the stem
        input_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
    };
    
    // Format: {base name}.clean.jsonl
    format!("{}.clean.jsonl", base_name)
}

fn main() {
    let test_cases = vec![
        "sponge-text-2024-08-00000.jsonl",
        "sponge-text-2024-08-00003",
        "sponge-text-2024-08-00004.jsonl.gz",
        "sponge-text-2024-08-00005.jsonl.zst",
    ];
    
    for test in test_cases {
        let path = PathBuf::from(test);
        println!("{} -> {}", test, get_purified_filename(&path));
    }
}
