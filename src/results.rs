
pub(crate) struct MLResult {
    name: String,
    acc: f32,
    mae: f32
}

impl MLResult {
    pub(crate) fn new(name: String, acc: f32, mae: f32) -> Self {
        MLResult {
            name,
            acc,
            mae
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
}
