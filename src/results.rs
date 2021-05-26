use std::fmt;
use std::fmt::Formatter;

pub(crate) struct MLResult {
    name: String,
    acc: f32,
    mae: f32,
    mse: f32,
}

impl MLResult {
    pub(crate) fn new(name: String, acc: f32, mae: f32) -> Self {
        MLResult {
            name,
            acc,
            mae,
            mse: 0.0
        }
    }

    pub(crate) fn name(&self) -> String {
        self.name.to_string()
    }

    pub(crate) fn acc(&self) -> f32 {
        self.acc
    }

    pub(crate) fn mae(&self) -> f32 {
        self.mae
    }

    pub(crate) fn mse(&self) -> f32 {
        self.mse
    }
}

impl fmt::Display for MLResult {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let s = format!("{} :: acc={}", self.name, self.acc);

        write!(f, "{}", s)
    }
}

impl Default for MLResult {
    fn default() -> Self {
        MLResult::new("--".to_string(), 0.0, 0.0)
    }
}