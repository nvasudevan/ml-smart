use crate::results::{best_k, KMeansResult};
use crate::kmeans::run_n_k;
use std::time::Instant;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::dataset::Dataset;
use rayon::prelude::{ParallelExtend, ParallelIterator, IntoParallelIterator};

pub(crate) fn run(ds: &Dataset<f32,f32>) {
    let dm = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );

    let labels_true = &ds.target;
    for i in 0..1 {
        let mut results: Vec<KMeansResult> = vec![];
        let start = Instant::now();
        let mut n = 5;
        loop {
            eprint!(".");
            results.par_extend(
                (3..4).into_par_iter().map( |k| run_n_k(&dm, n, k, labels_true))
            );
            n += 1;
            if n > 50 {
                break;
            }
        }

        println!("results: n={}", results.len());
        let end = start.elapsed();
        println!("[{}] time taken: {}", i, end.as_secs());
        best_k(results);
    }
}
