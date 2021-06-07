use smartcore::cluster::kmeans::{KMeans, KMeansParameters};
use smartcore::dataset::{Dataset, digits};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::metrics::{accuracy, mean_absolute_error, completeness_score, homogeneity_score};

use crate::dataset::DatasetParseError;
use crate::results::{MLResult, best_k, KMeansResult};
use std::time::Instant;
use rayon::prelude::{IntoParallelIterator, ParallelExtend, ParallelIterator};

use crate::kmeans::run_n_k;

const MAX_ITERATIONS: [usize;6] = [50, 75, 100, 125, 150, 175];

pub(crate) fn run(ds: &Dataset<f32,f32>) {
    let dm = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );

    let labels_true = &ds.target;
    for i in 0..3 {
        let mut results: Vec<KMeansResult> = vec![];
        let start = Instant::now();
        let mut n = 10;
        loop {
            eprint!(".");
            results.par_extend(
                (4..16).into_par_iter().map( |k| run_n_k(&dm, n, k, labels_true))
            );
            n += 5;
            if n > 100 {
                break;
            }
        }

        println!("results: n={}", results.len());
        let end = start.elapsed();
        println!("[{}] time taken: {}", i, end.as_secs());
        best_k(results);
    }

    // println!("v_measure score: {}", v_measure_score(&labels_true, &labels_pred));
}

