use std::fmt;

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

#[derive(Copy, Clone)]
struct DBScanParams<'a> {
    eps: f32,
    distance: &'a KNNDistance,
    algorithm: &'a KNNAlgorithmName,
}

impl<'a> DBScanParams<'a> {
    fn new(eps: f32, distance: &'a KNNDistance, algorithm: &'a KNNAlgorithmName) -> Self {
        Self {
            eps,
            distance,
            algorithm,
        }
    }

    fn inc_eps(&mut self, inc: f32) {
        self.eps += inc;
    }
}

impl<'a> fmt::Display for DBScanParams<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut params = Vec::<&str>::new();
        match self.distance {
            KNNDistance::Euclidean => { params.push("Euclidean"); }
            KNNDistance::Hamming => { params.push("Hamming"); }
            KNNDistance::Manhattan => { params.push("Manhattan"); }
        }

        match self.algorithm {
            KNNAlgorithmName::LinearSearch => { params.push("LinearSearch") }
            KNNAlgorithmName::CoverTree => { params.push("CoverTree") }
        }

        write!(f,
               "{}",
               format!("DBScan [eps={} {}]", self.eps, params.join(" + ")))
    }
}

struct DBScanRun<'a> {
    tt_ds: TrainTestDataset<'a>,
    pub(crate) results: Vec<MLResult>,
}

impl<'a> DBScanRun<'a> {
    fn new(ds: &'a Dataset<f32, f32>) -> Self {
        let tt_ds = TrainTestDataset::new(&ds);
        Self {
            tt_ds,
            results: Vec::<MLResult>::new(),
        }
    }

    fn add_result(&mut self, res: MLResult) {
        self.results.push(res)
    }

    fn predict(&mut self, algo_params: DBScanParams) -> Result<MLResult, DatasetParseError> {
        let p = match algo_params.distance {
            KNNDistance::Hamming => {
                let mut params = DBSCANParameters::default()
                    .with_distance(Distances::hamming())
                    .with_algorithm(algo_params.algorithm.clone())
                    .with_eps(algo_params.eps);
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
                    .with_algorithm(algo_params.algorithm.clone())
                    .with_eps(algo_params.eps);
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
                    .with_algorithm(algo_params.algorithm.clone())
                    .with_eps(algo_params.eps);
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

        let acc = accuracy(&self.tt_ds.ds.target, &p);
        let mae = mean_absolute_error(&self.tt_ds.ds.target, &p);

        let res = MLResult::new(algo_params.to_string(), acc, mae);
        Ok(res)
    }

    fn train_and_test(&mut self, params: DBScanParams) -> Result<(), DatasetParseError> {
        println!("\n=> running {}", params);
        let mut curr_res = MLResult::new(params.to_string(), 0.0, 0.0);
        let mut no_changes = 0;

        let mut curr_params = params.clone();
        loop {
            eprint!(".");
            let res = self.predict(curr_params)?;
            if res.acc() > curr_res.acc() {
                curr_res = res;
                no_changes = 0;
            } else {
                no_changes += 1;
            }
            if no_changes >= crate::MAX_NO_CHANGES {
                break;
            }

            curr_params.inc_eps(0.1);
        }
        self.add_result(curr_res);

        Ok(())
    }

    fn run(&mut self) -> Result<(), DatasetParseError> {
        let distances = [
            KNNDistance::Euclidean,
            KNNDistance::Hamming,
            KNNDistance::Manhattan
        ];
        let algos = [
            KNNAlgorithmName::CoverTree,
            KNNAlgorithmName::LinearSearch
        ];
        for dist in distances.iter() {
            for algo in algos.iter() {
                // kick off with eps=0.5 (default)
                let params = DBScanParams::new(0.5, dist, algo);
                let _ = self.train_and_test(params)?;
            }
        }

        Ok(())
    }
}

pub(crate) fn run(ds: &Dataset<f32, f32>) -> Result<Vec<MLResult>, DatasetParseError> {
    println!("\n=> Running DBScan on flag dataset ...");
    let mut dbscan_run = DBScanRun::new(ds);
    let _ = dbscan_run.run();

    Ok(dbscan_run.results)
}
