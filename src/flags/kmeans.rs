use smartcore::cluster::kmeans::{KMeans, KMeansParameters};
use smartcore::dataset::Dataset;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::metrics::{accuracy, mean_absolute_error};

use crate::dataset::DatasetParseError;
use crate::results::MLResult;

const MAX_ITERATIONS: [usize;5] = [100, 150, 200, 250, 300];
const START_K: usize = 2;

fn train_and_test(ds: &Dataset<f32, f32>, k: usize, max_iter: usize) -> Result<MLResult, DatasetParseError> {
    eprint!(".");
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );
    let mut params = KMeansParameters::default().with_k(k).with_max_iter(max_iter);
    let model = KMeans::fit(&nm_matrix, params)?;

    //now try on test data
    let p = model.predict(&nm_matrix)?;
    let params_s = format!("K-Means k={}, max_iter={}", k, max_iter);
    let res = MLResult::new(params_s,
                            accuracy(&ds.target, &p),
                            mean_absolute_error(&ds.target, &p),
    );

    Ok(res)
}

pub(crate) fn run(ds: &Dataset<f32, f32>) -> Result<Vec<MLResult>, DatasetParseError> {
    println!("=> Running kMeans on flag dataset ...");
    let mut results = Vec::<MLResult>::new();
    for max_iter in MAX_ITERATIONS.iter() {
        let mut k = START_K;
        let mut no_changes = 0;
        let mut curr_res = MLResult::default();
        loop {
            let res = train_and_test(ds, k, *max_iter)?;
            if res.acc() >= curr_res.acc() {
                no_changes = 0;
                curr_res = res;
            } else {
                no_changes += 1;
            }
            if no_changes >= crate::MAX_NO_CHANGES {
                break
            }
            k += 1;
        }
        results.push(curr_res);
    }

    Ok(results)
}

