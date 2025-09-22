use regex::Regex;

pub fn clean_text(text: &str, punctuation_chars: &str) -> String {
    let text = text.to_lowercase();

    // Replace punctuation with spaces based on configurable character set
    let punctuation_chars: Vec<char> = punctuation_chars.chars().collect();
    let text: String = text.chars().map(|c| {
        if punctuation_chars.contains(&c) {
            ' '
        } else {
            c
        }
    }).collect();

    // Replace multiple whitespace characters with a single space
    let re = Regex::new(r"\s+").unwrap();
    let text = re.replace_all(&text, " ").to_string();

    text.trim().to_string()
}

pub fn default_punctuation_chars() -> String {
    // Unicode character mapping:
    // \u{201c} = " (left double quotation mark)
    // \u{201d} = " (right double quotation mark)
    // \u{2018} = ' (left single quotation mark)
    // \u{2019} = ' (right single quotation mark)
    // \u{2014} = — (em dash)

    // CONSIDER: using an allow list of alphanumeric characters instead of a remove list
    // of punctuation. This would be more maintainable and predictable.
    // CONSIDER: ×

    "!'\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~\u{201c}\u{201d}\u{2018}\u{2019}\u{2014}".to_string() // Default punctuation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_text_with_default_punctuation() {
        let input = "  Hello, world! This is a test.  ";
        let expected = "hello world this is a test";
        let punctuation = default_punctuation_chars();
        assert_eq!(clean_text(input, &punctuation), expected);
    }

    #[test]
    fn test_clean_text_with_custom_punctuation() {
        let input = "Remove|these|character!s";
        let expected = "remove these character!s";
        let punctuation = "|";
        assert_eq!(clean_text(input, punctuation), expected);
    }

    #[test]
    fn test_clean_text_multiple_whitespace() {
        let input = "lots   of   whitespace";
        let expected = "lots of whitespace";
        let punctuation = default_punctuation_chars();
        assert_eq!(clean_text(input, &punctuation), expected);
    }

    #[test]
    fn test_clean_text_uppercase() {
        let input = "UPPERCASE";
        let expected = "uppercase";
        let punctuation = default_punctuation_chars();
        assert_eq!(clean_text(input, &punctuation), expected);
    }

    #[test]
    fn test_clean_text_empty_string() {
        let input = "";
        let expected = "";
        let punctuation = default_punctuation_chars();
        assert_eq!(clean_text(input, &punctuation), expected);
    }

    #[test]
    fn test_clean_text_no_changes() {
        let input = "this should not change";
        let expected = "this should not change";
        let punctuation = default_punctuation_chars();
        assert_eq!(clean_text(input, &punctuation), expected);
    }
}
