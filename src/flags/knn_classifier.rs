use smartcore::algorithm::neighbour::KNNAlgorithmName;
use smartcore::dataset::Dataset;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::math::distance::{
    Distances,
    euclidian::Euclidian,
    hamming::Hamming,
    manhattan::Manhattan,
};
use smartcore::metrics::{accuracy, mean_absolute_error};
use smartcore::model_selection::train_test_split;
use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};
use smartcore::neighbors::knn_regressor::KNNRegressor;
use smartcore::neighbors::KNNWeightFunction;

use crate::dataset::DatasetParseError;
use crate::flags::knn_regression;
use crate::results::MLResult;

pub(crate) struct KNNClassifierRun<'a> {
    pub(crate) results: Vec<MLResult>,
    ds: &'a Dataset<f32, f32>,
    x_train: DenseMatrix<f32>,
    y_train: Vec<f32>,
    x_test: DenseMatrix<f32>,
    y_test: Vec<f32>,
}

#[derive(Clone, Copy)]
enum KNNDistance {
    Euclidean,
    Hamming,
    Manhattan,
}

fn knn_params_as_str(distance: &KNNDistance, algo: &KNNAlgorithmName, weight: &KNNWeightFunction) -> Vec<String> {
    let mut tags = Vec::<String>::new();
    match distance {
        KNNDistance::Hamming => {
            tags.push("Hamming".to_string());
        }
        KNNDistance::Manhattan => {
            tags.push("Manhattan".to_string());
        }
        _ => {
            tags.push("Euclidean".to_string());
        }
    };

    match algo {
        KNNAlgorithmName::LinearSearch => {
            tags.push("LinearSearch".to_string());
        }
        _ => {
            tags.push("CoverTree".to_string());
        }
    }

    match weight {
        KNNWeightFunction::Distance => {
            tags.push("Inverse".to_string());
        }
        _ => {
            tags.push("Uniform".to_string());
        }
    }

    tags
}

impl<'a> KNNClassifierRun<'a> {
    fn new(ds: &'a Dataset<f32, f32>) -> Self {
        let nm_matrix = DenseMatrix::from_array(
            ds.num_samples, ds.num_features, &ds.data,
        );
        let (x_train,
            x_test,
            y_train,
            y_test) = train_test_split(
            &nm_matrix,
            &ds.target,
            crate::TRAINING_TEST_SIZE_RATIO,
            true,
        );

        Self {
            results: Vec::<MLResult>::new(),
            ds,
            x_train,
            y_train,
            x_test,
            y_test,
        }
    }

    fn add_result(&mut self, res: MLResult) {
        self.results.push(res)
    }

    fn predict(&mut self,
               k: usize,
               distance: KNNDistance,
               algorithm: &KNNAlgorithmName
    ) -> Result<MLResult, DatasetParseError> {
        let mut tags = Vec::<String>::new();
        let p = match distance {
            KNNDistance::Hamming => {
                let mut params = KNNClassifierParameters::default()
                    .with_distance(Distances::hamming())
                    .with_algorithm(algorithm.clone())
                    .with_k(k);
                // let params_algo = params.with_algorithm(algorithm.clone());
                // let params_k = params.with_k(k);
                tags.append(&mut knn_params_as_str(
                    &KNNDistance::Hamming,
                    &algorithm, &KNNWeightFunction::Uniform,
                ));
                let logr = KNNClassifier::fit(
                    &self.x_train, &self.y_train, params,
                )?;
                logr.predict(&self.x_test)?
            }
            KNNDistance::Manhattan => {
                let params = KNNClassifierParameters::default()
                    .with_distance(Distances::manhattan())
                    .with_algorithm(algorithm.clone())
                    .with_k(k);
                // let params_algo = params.with_algorithm(algorithm.clone());
                tags.append(&mut knn_params_as_str(
                    &KNNDistance::Manhattan,
                    &algorithm, &KNNWeightFunction::Uniform,
                ));
                let logr = KNNClassifier::fit(
                    &self.x_train, &self.y_train, params,
                )?;
                logr.predict(&self.x_test)?
            }
            _ => {
                let params = KNNClassifierParameters::default()
                    .with_algorithm(algorithm.clone())
                    .with_k(k);
                // let params_algo = params.with_algorithm(algorithm.clone());
                tags.append(&mut knn_params_as_str(
                    &KNNDistance::Euclidean,
                    &algorithm, &KNNWeightFunction::Uniform,
                ));
                let logr = KNNClassifier::fit(
                    &self.x_train, &self.y_train, params,
                )?;
                logr.predict(&self.x_test)?
            }
        };

        let params_s = format!("kNN, k={} [{}]", k, tags.join(", "));
        let acc = accuracy(&self.y_test, &p);
        let mae = mean_absolute_error(&self.y_test, &p);

        let res = MLResult::new(params_s, acc, mae);
        Ok(res)
    }

    fn train_and_test(&mut self, distance: KNNDistance, algorithm: &KNNAlgorithmName)
                      -> Result<(), DatasetParseError> {

        let mut curr_res = MLResult::default();
        let mut no_changes = 0;
        // start from default value of k=3.
        let mut curr_k = 3;

        loop {
            let res = self.predict(curr_k, distance, algorithm)?;
            if res.acc() >= curr_res.acc() {
                no_changes = 0;
                curr_res = res;
            } else {
                no_changes += 1;
            }
            if no_changes >= crate::MAX_NO_CHANGES {
                break
            }
            curr_k += 1;
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

    fn run_algorithm(&mut self, weight: KNNWeightFunction) -> Result<(), DatasetParseError> {
        // run with default algorithm (CoverTree)
        self.run_distance(KNNAlgorithmName::CoverTree)?;
        // run with LinearSearch
        self.run_distance(KNNAlgorithmName::LinearSearch)?;

        Ok(())
    }

    fn run_weight(&mut self) -> Result<(), DatasetParseError> {
        // run with default weight (Uniform)
        self.run_algorithm(KNNWeightFunction::Uniform)?;
        // run the distance, which uses inverse function
        self.run_algorithm(KNNWeightFunction::Distance)?;

        Ok(())
    }
}

pub(crate) fn run(ds: &Dataset<f32, f32>) -> Result<Vec<MLResult>, DatasetParseError> {
    println!("\n=> Running knn classifier on flag dataset ...");
    let mut knn_class_run = KNNClassifierRun::new(ds);
    let _ = knn_class_run.run_weight()?;

    Ok(knn_class_run.results)
}
