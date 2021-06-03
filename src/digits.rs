use std::fmt;
use std::fmt::Formatter;
use std::time::{Duration, Instant};

use rayon::prelude::{IntoParallelIterator, ParallelExtend, ParallelIterator};
use smartcore::cluster::kmeans::{KMeans, KMeansParameters};
use smartcore::dataset::{Dataset, digits};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::metrics::{completeness_score, homogeneity_score, v_measure_score};

use crate::results::{KMeansResult, best_k};

fn run_n_k(dm: &DenseMatrix<f32>, n: usize, k: usize, labels_true: &Vec<f32>) -> KMeansResult {
    let model = KMeans::fit(
        dm, KMeansParameters::default()
            .with_k(k).with_max_iter(n),
    ).expect("Failed to create a model");
    let labels_pred = model.predict(dm)
        .expect("Unable to predict");

    KMeansResult::new(n, k,
                      homogeneity_score(labels_true, &labels_pred),
                      completeness_score(labels_true, &labels_pred),
    )
}

pub(crate) fn run() {
    let digits_ds = digits::load_dataset();
    let dm = DenseMatrix::from_array(
        digits_ds.num_samples, digits_ds.num_features, &digits_ds.data,
    );

    let labels_true = digits_ds.target;
    for i in 0..10 {
        let mut results: Vec<KMeansResult> = vec![];
        let start = Instant::now();
        for n in [50, 75, 100, 125, 150, 175].iter() {
            eprint!(".");
            results.par_extend(
                (6..16).into_par_iter().map( |k| run_n_k(&dm, *n, k, &labels_true))
            );
            // results.extend(
            //     (6..16).into_iter().map(|k| run_n_k(&dm, *n, k, &labels_true))
            // );
        }

        println!("results: n={}", results.len());
        let end = start.elapsed();
        println!("[{}] time taken: {}", i, end.as_secs());
        best_k(results);
    }

    // println!("v_measure score: {}", v_measure_score(&labels_true, &labels_pred));
}
