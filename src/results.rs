use std::fmt;
use std::fmt::Formatter;
use prettytable::Table;

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

    pub(crate) fn set_name(&mut self, name: String) {
        self.name = name
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

pub(crate) fn show(results: Vec<MLResult>) {
    let mut table = Table::new();
    table.add_row(row!["ML algo", "accuracy", "MAE", "MSE*"]);
    for ml in results {
        table.add_row(row![ml.name(), ml.acc(), ml.mae(), ml.mse()]);
    }

    table.printstd();
    println!();
}

pub(crate) struct KMeansResult {
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) h_score: f32,
    pub(crate) c_score: f32,
}

impl fmt::Display for KMeansResult {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let s = format!(
            "(n,k,hom,com): ({},{},{},{})",
            self.n, self.k, self.h_score, self.c_score
        );

        writeln!(f, "{}", s)
    }
}

impl KMeansResult {
    pub(crate) fn new(n: usize, k: usize, h_score: f32, c_score: f32) -> Self {
        Self {
            n,
            k,
            h_score,
            c_score,
        }
    }
}
pub(crate) fn best_k(results: Vec<KMeansResult>) {
    let mut best_hom_n = 0;
    let mut best_hom_k = 0;
    let mut best_h_score = 0.0;
    let mut best_com_n = 0;
    let mut best_com_k = 0;
    let mut best_c_score = 0.0;
    for res in results {
        if res.h_score > best_h_score {
            best_h_score = res.h_score;
            best_hom_k = res.k;
            best_hom_n = res.n;
        }
        if res.c_score > best_c_score {
            best_c_score = res.c_score;
            best_com_k = res.k;
            best_com_n = res.n;
        }
    }
    println!("best (hom, n, K): ({},{},{}) (com, n, K): ({},{},{})",
             best_h_score, best_hom_n, best_hom_k,
             best_c_score, best_com_n, best_com_k);
}

