use anyhow::{Error, Result};
use serde_json::Value;

pub fn get_nested_json_val(obj: &Value, key: &String) -> Result<String, Error> {
    let mut current = obj;
    for subkey in key.split('.') {
        current = current
            .get(subkey)
            .ok_or_else(|| anyhow::anyhow!("Key '{}' not found in JSON object", subkey))?;
    }

    current
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Value at key '{}' is not a string", key))
        .map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_get_nested_json_val_simple_key() {
        let obj = json!({
            "text": "Hello, world!",
            "id": 123
        });
        
        let result = get_nested_json_val(&obj, &"text".to_string()).unwrap();
        assert_eq!(result, "Hello, world!");
    }

    #[test]
    fn test_get_nested_json_val_nested_key() {
        let obj = json!({
            "user": {
                "name": "John Doe",
                "details": {
                    "email": "john@example.com"
                }
            }
        });
        
        let result = get_nested_json_val(&obj, &"user.name".to_string()).unwrap();
        assert_eq!(result, "John Doe");
        
        let result = get_nested_json_val(&obj, &"user.details.email".to_string()).unwrap();
        assert_eq!(result, "john@example.com");
    }

    #[test]
    fn test_get_nested_json_val_missing_key() {
        let obj = json!({
            "text": "Hello"
        });
        
        let result = get_nested_json_val(&obj, &"missing".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Key 'missing' not found"));
    }

    #[test]
    fn test_get_nested_json_val_missing_nested_key() {
        let obj = json!({
            "user": {
                "name": "John"
            }
        });
        
        let result = get_nested_json_val(&obj, &"user.email".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Key 'email' not found"));
    }

    #[test]
    fn test_get_nested_json_val_non_string_value() {
        let obj = json!({
            "count": 42,
            "active": true
        });
        
        let result = get_nested_json_val(&obj, &"count".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Value at key 'count' is not a string"));
        
        let result = get_nested_json_val(&obj, &"active".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Value at key 'active' is not a string"));
    }

    #[test]
    fn test_get_nested_json_val_empty_string() {
        let obj = json!({
            "empty": ""
        });
        
        let result = get_nested_json_val(&obj, &"empty".to_string()).unwrap();
        assert_eq!(result, "");
    }
}