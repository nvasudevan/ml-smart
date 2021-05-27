use smartcore::algorithm::neighbour::KNNAlgorithmName;
use smartcore::cluster::dbscan::{DBSCAN, DBSCANParameters};
use smartcore::dataset::Dataset;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::math::distance::Distances;
use smartcore::metrics::{accuracy, mean_absolute_error};
use smartcore::model_selection::train_test_split;

use crate::{algo_params_as_str, KNNDistance};
use crate::dataset::{DatasetParseError, TrainTestDataset};
use crate::results::MLResult;

struct DBScanRun<'a> {
    tt_ds: TrainTestDataset<'a>,
    pub(crate) results: Vec<MLResult>,
}

impl<'a> DBScanRun<'a> {
    fn new(ds: &'a Dataset<f32, f32>) -> Self {
       let tt_ds = TrainTestDataset::new(&ds);
        Self {
            tt_ds,
            results: Vec::<MLResult>::new()
        }
    }

    fn add_result(&mut self, res: MLResult) {
        self.results.push(res)
    }

    fn predict(&mut self,
               eps: f32,
               distance: KNNDistance,
               algorithm: &KNNAlgorithmName
    ) -> Result<MLResult, DatasetParseError> {
        let mut tags = Vec::<String>::new();
        let p = match distance {
            KNNDistance::Hamming => {
                let mut params = DBSCANParameters::default()
                    .with_distance(Distances::hamming())
                    .with_algorithm(algorithm.clone())
                    .with_eps(eps);
                // let params_algo = params.with_algorithm(algorithm.clone());
                // let params_k = params.with_k(k);
                tags.append(&mut algo_params_as_str(
                    &KNNDistance::Hamming,
                    &algorithm, None
                ));
                let nm_matrix = DenseMatrix::from_array(
                    self.tt_ds.ds.num_samples,
                    self.tt_ds.ds.num_features,
                    &self.tt_ds.ds.data,
                );
                let model = DBSCAN::fit(
                    &nm_matrix, params,
                )?;
                model.predict(&nm_matrix)?
            }
            KNNDistance::Manhattan => {
                let params = DBSCANParameters::default()
                    .with_distance(Distances::manhattan())
                    .with_algorithm(algorithm.clone())
                    .with_eps(eps);
                tags.append(&mut algo_params_as_str(
                    &KNNDistance::Manhattan,
                    &algorithm, None,
                ));
                let nm_matrix = DenseMatrix::from_array(
                    self.tt_ds.ds.num_samples,
                    self.tt_ds.ds.num_features,
                    &self.tt_ds.ds.data,
                );
                let model = DBSCAN::fit(
                    &nm_matrix, params,
                )?;
                model.predict(&nm_matrix)?
            }
            _ => {
                let params = DBSCANParameters::default()
                    .with_algorithm(algorithm.clone())
                    .with_eps(eps);
                tags.append(&mut algo_params_as_str(
                    &KNNDistance::Euclidean,
                    &algorithm, None
                ));
                let nm_matrix = DenseMatrix::from_array(
                    self.tt_ds.ds.num_samples,
                    self.tt_ds.ds.num_features,
                    &self.tt_ds.ds.data,
                );
                let model = DBSCAN::fit(
                    &nm_matrix, params,
                )?;
                model.predict(&nm_matrix)?
            }
        };

        let params_s = format!("DBScan, eps={} [{}]", eps, tags.join(", "));
        let acc = accuracy(&self.tt_ds.ds.target, &p);
        let mae = mean_absolute_error(&self.tt_ds.ds.target, &p);

        let res = MLResult::new(params_s, acc, mae);
        Ok(res)
    }

    fn train_and_test(&mut self, distance: KNNDistance, algorithm: &KNNAlgorithmName)
                      -> Result<(), DatasetParseError> {

        let mut curr_res = MLResult::default();
        let mut no_changes = 0;
        // start from default value of 0.5
        let mut eps = 0.5;
        curr_res.set_name(format!("DBScan eps={}", eps));

        loop {
            // eprint!(".");
            let res = self.predict(eps, distance, algorithm)?;
            println!("[{}], res: {}", curr_res.acc(), res);
            if res.acc() > curr_res.acc() {
                curr_res = res;
                no_changes = 0;
            } else {
                no_changes += 1;
            }
            if no_changes >= crate::MAX_NO_CHANGES {
                break
            }
            eps += 0.1;
        }
        self.add_result(curr_res);

        Ok(())
    }

    fn run_distance(&mut self, algorithm: KNNAlgorithmName) -> Result<(), DatasetParseError> {
        let _ = self.train_and_test(
            KNNDistance::Euclidean,
            &algorithm,
        )?;

        let _ = self.train_and_test(
            KNNDistance::Hamming,
            &algorithm,
        )?;

        let _ = self.train_and_test(
            KNNDistance::Manhattan,
            &algorithm,
        )?;

        Ok(())
    }

    fn run_algorithm(&mut self) -> Result<(), DatasetParseError> {
        // run with default algorithm (CoverTree)
        self.run_distance(KNNAlgorithmName::CoverTree)?;
        // run with LinearSearch
        self.run_distance(KNNAlgorithmName::LinearSearch)?;

        Ok(())
    }
}

pub(crate) fn run(ds: &Dataset<f32, f32>) -> Result<Vec<MLResult>, DatasetParseError> {
    println!("\n=> Running DBScan on flag dataset ...");
    let mut dbscan_run = DBScanRun::new(ds);
    let _ = dbscan_run.run_algorithm()?;

    Ok(dbscan_run.results)
}
