use std::{
    io,
    fmt,
    num::ParseFloatError
};
use std::fmt::Formatter;

pub(crate) mod wine;

pub(crate) const WINE_DATASET: &str = "./datasets/wine.data";

#[derive(Debug)]
pub(crate) struct DatasetParseError {
    pub(crate) msg: String
}

impl From<io::Error> for DatasetParseError {
    fn from(err: io::Error) -> Self {
        Self {
            msg: err.to_string()
        }
    }
}

impl From<ParseFloatError> for DatasetParseError {
    fn from(err: ParseFloatError) -> Self {
        Self {
            msg: err.to_string()
        }
    }
}

impl fmt::Display for DatasetParseError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let s = format!("error: {}", self.msg);
        write!(f, "{}", s)
    }
}