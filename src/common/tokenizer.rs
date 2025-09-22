use anyhow::{Error, Result};
use tiktoken_rs::{cl100k_base, o200k_base, p50k_base, p50k_edit, r50k_base, CoreBPE};
use unicode_segmentation::UnicodeSegmentation;

pub struct OmniTokenizer {
    pub tokenizer_name: String,
    pub inner: Option<CoreBPE>,
}

fn get_tiktoken_tokenizer(name: &str) -> Option<CoreBPE> {
    match name {
        "r50k" => Some(r50k_base().expect("Failed to initialize r50k tokenizer")),
        "p50k" => Some(p50k_base().expect("Failed to initialize p50k tokenizer")),
        "p50k_edit" => Some(p50k_edit().expect("Failed to initialize p50k_edit tokenizer")),
        "cl100k" => Some(cl100k_base().expect("Failed to initialize cl100k tokenizer")),
        "o200k" => Some(o200k_base().expect("Failed to initialize o200k tokenizer")),
        _ => None,
    }
}

impl OmniTokenizer {
    pub fn new(tokenizer_name: &str) -> Result<Self, Error> {
        let inner_tokenizer = match tokenizer_name {
            name if get_tiktoken_tokenizer(name).is_some() => get_tiktoken_tokenizer(name),
            "uniseg" => None,
            _ => {
                println!("Warning: Unknown tokenizer '{}', using character-level tokenization", tokenizer_name);
                None
            }
        };
        Ok(OmniTokenizer {
            tokenizer_name: tokenizer_name.to_string(),
            inner: inner_tokenizer,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        // If we have a BPE tokenizer, use it
        if let Some(ref tokenizer) = self.inner {
            return tokenizer.encode_ordinary(text);
        }

        // Otherwise, use custom tokenization
        match self.tokenizer_name.as_str() {
            "uniseg" => text
                .split_word_bounds()
                .map(|w| {
                    use std::hash::{DefaultHasher, Hash, Hasher};
                    let mut hasher = DefaultHasher::new();
                    w.hash(&mut hasher);
                    hasher.finish() as usize
                })
                .collect(),
            _ => {
                // Default to character level for unknown tokenizers
                text.bytes().map(|b| b as usize).collect()
            }
        }
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        // If we have a BPE tokenizer, use it
        if let Some(ref tokenizer) = self.inner {
            return tokenizer.decode(tokens.to_vec()).unwrap_or_else(|_| {
                format!("(decode failed for tokens: {:?})", tokens)
            });
        }

        // Otherwise, handle custom tokenization
        match self.tokenizer_name.as_str() {
            "uniseg" => {
                // Can't decode back from hashes to text
                format!("(uniseg tokens: {:?})", tokens)
            }
            _ => {
                // For character level, convert bytes back to string
                let bytes: Vec<u8> = tokens.iter().map(|&t| t as u8).collect();
                String::from_utf8(bytes).unwrap_or_else(|_| {
                    format!("(decode failed for tokens: {:?})", tokens)
                })
            }
        }
    }

    /// When counting tokens for filters, we omit space tokens as uninformative for
    /// applying thresholds, e.g. minimum non-space tokens to index an eval instance.
    pub fn is_space_token(&self, token_id: usize) -> bool {
        match self.tokenizer_name.as_str() {
            "cl100k" => {
                // Actual whitespace tokens for cl100k tokenizer (verified by decoding)
                // Includes tabs, newlines, spaces, and various whitespace characters
                matches!(token_id,
                    197 | 198 | 199 | 200 | 201 | 216 | 217 | 218 | 219 | 220 |
                    256 | 257 | 260 | 262 | 271 | 286 | 298 | 310 | 319 | 338 |
                    394 | 415 | 465 | 504 | 573 | 667 | 692 | 720 | 792 | 881 |
                    996 | 1014 | 1038 | 1078 | 1084 | 1164 | 1408 | 1432 | 1602 | 1696 |
                    1733 | 1827 | 1835 | 1881 | 1961
                )
            }
            "o200k" => {
                // Actual whitespace tokens for o200k tokenizer (verified by decoding)
                matches!(token_id,
                    197 | 198 | 199 | 200 | 201 | 216 | 217 | 218 | 219 | 220 |
                    256 | 257 | 269 | 271 | 279 | 309 | 335 | 352 | 370 | 408 |
                    506 | 530 | 626 | 699 | 793 | 833 | 968 | 983 | 1202 | 1213 |
                    1397 | 1414 | 1686 | 1698 | 1699 | 1944 | 1999
                )
            }
            "p50k" | "p50k_edit" => {
                // Actual whitespace tokens for p50k family tokenizers (verified by decoding)
                matches!(token_id,
                    197 | 198 | 199 | 200 | 201 | 216 | 217 | 218 | 219 | 220 |
                    628 | 1849
                )
            }
            "r50k" => {
                // Actual whitespace tokens for r50k tokenizer (verified by decoding)
                matches!(token_id,
                    197 | 198 | 199 | 200 | 201 | 216 | 217 | 218 | 219 | 220 |
                    628 | 1849
                )
            }
            "uniseg" => {
                // uniseg tokenizes at word boundaries, spaces are not separate tokens
                false
            }
            _ => {
                // For unknown tokenizers, try to decode if we have a BPE tokenizer
                if let Some(ref tokenizer) = self.inner {
                    if let Ok(text) = tokenizer.decode(vec![token_id]) {
                        text.trim().is_empty()
                    } else {
                        false
                    }
                } else {
                    // No tokenizer - for character-level fallback, check if it's a whitespace byte
                    matches!(token_id as u8, b' ' | b'\t' | b'\n' | b'\r')
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_new() {
        // Test all tiktoken tokenizers
        let r50k = OmniTokenizer::new("r50k").unwrap();
        assert_eq!(r50k.tokenizer_name, "r50k");
        assert!(r50k.inner.is_some());


        let p50k = OmniTokenizer::new("p50k").unwrap();
        assert_eq!(p50k.tokenizer_name, "p50k");
        assert!(p50k.inner.is_some());

        let p50k_edit = OmniTokenizer::new("p50k_edit").unwrap();
        assert_eq!(p50k_edit.tokenizer_name, "p50k_edit");
        assert!(p50k_edit.inner.is_some());

        let cl100k = OmniTokenizer::new("cl100k").unwrap();
        assert_eq!(cl100k.tokenizer_name, "cl100k");
        assert!(cl100k.inner.is_some());

        let o200k = OmniTokenizer::new("o200k").unwrap();
        assert_eq!(o200k.tokenizer_name, "o200k");
        assert!(o200k.inner.is_some());

        // Test custom tokenizers
        let uniseg = OmniTokenizer::new("uniseg").unwrap();
        assert_eq!(uniseg.tokenizer_name, "uniseg");
        assert!(uniseg.inner.is_none()); // uniseg doesn't use BPE

        // Test unknown tokenizer fallback to character-level
        let unknown = OmniTokenizer::new("unknown").unwrap();
        assert_eq!(unknown.tokenizer_name, "unknown");
        assert!(unknown.inner.is_none()); // Unknown tokenizers use char-level
    }

    #[test]
    fn test_encode() {
        // Test BPE tokenizers
        let p50k = OmniTokenizer::new("p50k").unwrap();
        assert_eq!(p50k.encode("hello world"), vec![31373, 995]);

        let cl100k = OmniTokenizer::new("cl100k").unwrap();
        assert_eq!(cl100k.encode("hello world"), vec![15339, 1917]);

        let o200k = OmniTokenizer::new("o200k").unwrap();
        let o200k_tokens = o200k.encode("hello world");
        assert_eq!(o200k_tokens.len(), 2); // o200k should produce 2 tokens for "hello world"

        // Test custom tokenizers
        let uniseg = OmniTokenizer::new("uniseg").unwrap();
        let uniseg_tokens = uniseg.encode("hello world");
        assert_eq!(uniseg_tokens.len(), 3); // "hello", " ", "world"

        // Test unknown tokenizer fallback (character-level)
        let unknown = OmniTokenizer::new("unknown").unwrap();
        assert_eq!(unknown.encode("abc"), vec![97, 98, 99]); // ASCII values
    }

    #[test]
    fn test_is_space_token() {
        // Test cl100k
        let cl100k = OmniTokenizer::new("cl100k").unwrap();
        let space_token_cl = cl100k.encode(" ")[0];
        assert!(cl100k.is_space_token(space_token_cl));
        assert!(!cl100k.is_space_token(100)); // 100 is arbitrary non-space token

        // Test o200k
        let o200k = OmniTokenizer::new("o200k").unwrap();
        let space_token_o200k = o200k.encode(" ")[0];
        assert!(o200k.is_space_token(space_token_o200k));

        // Test p50k
        let p50k = OmniTokenizer::new("p50k").unwrap();
        let space_token_p50k = p50k.encode(" ")[0];
        assert!(p50k.is_space_token(space_token_p50k));
        assert!(!p50k.is_space_token(100)); // 100 is arbitrary non-space token

        // Test r50k
        let r50k = OmniTokenizer::new("r50k").unwrap();
        let space_token_r50k = r50k.encode(" ")[0];
        assert!(r50k.is_space_token(space_token_r50k));

        // Test uniseg - should always return false
        let uniseg = OmniTokenizer::new("uniseg").unwrap();
        assert!(!uniseg.is_space_token(0));
        assert!(!uniseg.is_space_token(12345));

        // Test unknown tokenizer fallback (character-level)
        let unknown = OmniTokenizer::new("unknown").unwrap();
        assert!(unknown.is_space_token(32)); // ASCII space
        assert!(!unknown.is_space_token(65)); // 'A'
    }
}
