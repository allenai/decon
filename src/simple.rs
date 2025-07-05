use anyhow::{Error, Result};
use crate::Config;

pub fn contamination_detect(config: &Config) -> Result<(), Error> {
    println!("TODO: implement simple contamination detection");
    println!("Config received: {:?}", config.mode);
    Ok(())
}