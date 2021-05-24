use smartcore::algorithm::neighbour::KNNAlgorithmName;
use smartcore::dataset::Dataset;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::math::distance::{Distance, Distances};
use smartcore::math::distance::euclidian::Euclidian;
use smartcore::math::distance::hamming::Hamming;
use smartcore::math::num::RealNumber;
use smartcore::metrics::{accuracy, mean_absolute_error};
use smartcore::model_selection::train_test_split;
use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};
use smartcore::neighbors::knn_regressor::KNNRegressor;

use crate::dataset::DatasetParseError;
use crate::results::MLResult;
use smartcore::math::distance::manhattan::Manhattan;
use smartcore::neighbors::KNNWeightFunction;

fn train_and_test_euclidean
(
    x_train: &DenseMatrix<f32>,
    y_train: &Vec<f32>,
    x_test: &DenseMatrix<f32>,
    y_test: &Vec<f32>,
    params: KNNClassifierParameters<f32, Euclidian>,
) -> Result<MLResult, DatasetParseError> {
    let logr = KNNClassifier::fit(
        x_train, y_train, params,
    )?;

    //now try on test data
    let p = logr.predict(x_test)?;
    let res = MLResult::new("kNN Classifier (Euclidean)".to_string(),
                            accuracy(y_test, &p),
                            mean_absolute_error(y_test, &p),
    );

    Ok(res)
}

fn train_and_test_hamming
(
    x_train: &DenseMatrix<f32>,
    y_train: &Vec<f32>,
    x_test: &DenseMatrix<f32>,
    y_test: &Vec<f32>,
    params: KNNClassifierParameters<f32, Hamming>,
) -> Result<MLResult, DatasetParseError> {
    let logr = KNNClassifier::fit(
        x_train, y_train, params,
    )?;

    //now try on test data
    let p = logr.predict(x_test)?;
    let res = MLResult::new("kNN Classifier (Hamming)".to_string(),
                            accuracy(y_test, &p),
                            mean_absolute_error(y_test, &p),
    );

    Ok(res)
}

fn train_and_test_manhattan
(
    x_train: &DenseMatrix<f32>,
    y_train: &Vec<f32>,
    x_test: &DenseMatrix<f32>,
    y_test: &Vec<f32>,
    params: KNNClassifierParameters<f32, Manhattan>,
) -> Result<MLResult, DatasetParseError> {
    let logr = KNNClassifier::fit(
        x_train, y_train, params,
    )?;

    //now try on test data
    let p = logr.predict(x_test)?;
    let res = MLResult::new("kNN Classifier (Manhattan)".to_string(),
                            accuracy(y_test, &p),
                            mean_absolute_error(y_test, &p),
    );

    Ok(res)
}

fn run_distance(ds: &Dataset<f32, f32>)
    -> Result<(), DatasetParseError> {

    let mut params: KNNClassifierParameters<f32, Euclidian> = KNNClassifierParameters::default();
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
    let euc_res = train_and_test_euclidean(&x_train, &y_train, &x_test, &y_test, params.clone())?;
    println!("[Euclidean + CoverTree + Uniform] accuracy: {}", euc_res.acc());

    let hamming_params = params.with_distance(Distances::hamming());
    let ham_res = train_and_test_hamming(&x_train, &y_train, &x_test, &y_test, hamming_params.clone())?;
    println!("[Hamming + CoverTree + Uniform] accuracy: {}", ham_res.acc());

    let manhat_params = hamming_params.with_distance(Distances::manhattan());
    let manhat_res = train_and_test_manhattan(&x_train, &y_train, &x_test, &y_test, manhat_params)?;
    println!("[Manhattan + CoverTree + Uniform] accuracy: {}", manhat_res.acc());

    Ok(())
}

fn run_algorithm(ds: &Dataset<f32, f32>) -> Result<(), DatasetParseError> {
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

    let mut params: KNNClassifierParameters<f32, Euclidian> = KNNClassifierParameters::default();
    let ls_params = params.with_algorithm(KNNAlgorithmName::LinearSearch);
    let euc_ls_res = train_and_test_euclidean(&x_train, &y_train, &x_test, &y_test, ls_params.clone())?;
    println!("[Euclidean + LinearSearch + Uniform] accuracy: {}", euc_ls_res.acc());

    let hamming_ls_params = ls_params.with_distance(Distances::hamming());
    let ham_ls_res = train_and_test_hamming(&x_train, &y_train, &x_test, &y_test, hamming_ls_params.clone())?;
    println!("[Hamming + LinearSearch + Uniform] accuracy: {}", ham_ls_res.acc());

    let manhat_ls_params = hamming_ls_params.with_distance(Distances::manhattan());
    let manhat_ls_res = train_and_test_manhattan(&x_train, &y_train, &x_test, &y_test, manhat_ls_params)?;
    println!("[Manhattan + LinearSearch + Uniform] accuracy: {}", manhat_ls_res.acc());

    Ok(())
}

fn run_weight(ds: &Dataset<f32, f32>) -> Result<(), DatasetParseError> {
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

    let mut params: KNNClassifierParameters<f32, Euclidian> = KNNClassifierParameters::default();
    let inv_params = params.with_weight(KNNWeightFunction::Distance);
    let euc_inv_res = train_and_test_euclidean(&x_train, &y_train, &x_test, &y_test, inv_params.clone())?;
    println!("[Euclidean + CoverTree + Inverse] accuracy: {}", euc_inv_res.acc());

    let hamming_inv_params = inv_params.clone().with_distance(Distances::hamming());
    let ham_inv_res = train_and_test_hamming(&x_train, &y_train, &x_test, &y_test, hamming_inv_params.clone())?;
    println!("[Hamming + CoverTree + Inverse] accuracy: {}", ham_inv_res.acc());

    let manhat_inv_params = hamming_inv_params.with_distance(Distances::manhattan());
    let manhat_inv_res = train_and_test_manhattan(&x_train, &y_train, &x_test, &y_test, manhat_inv_params)?;
    println!("[Manhattan + CoverTree + Inverse] accuracy: {}", manhat_inv_res.acc());

    let inv_ls_params = inv_params.with_algorithm(KNNAlgorithmName::LinearSearch);
    let euc_inv_ls_res = train_and_test_euclidean(&x_train, &y_train, &x_test, &y_test, inv_ls_params.clone())?;
    println!("[Euclidean + LinearSearch + Inverse] accuracy: {}", euc_inv_ls_res.acc());

    let ham_inv_ls_params = inv_ls_params.with_algorithm(KNNAlgorithmName::LinearSearch);
    let ham_inv_ls_res = train_and_test_euclidean(&x_train, &y_train, &x_test, &y_test, ham_inv_ls_params.clone())?;
    println!("[Hamming + LinearSearch + Inverse] accuracy: {}", ham_inv_ls_res.acc());

    let manhat_inv_ls_params = ham_inv_ls_params.with_algorithm(KNNAlgorithmName::LinearSearch);
    let manhat_inv_ls_res = train_and_test_euclidean(&x_train, &y_train, &x_test, &y_test, manhat_inv_ls_params.clone())?;
    println!("[Manhattan + LinearSearch + Inverse] accuracy: {}", manhat_inv_ls_res.acc());

    Ok(())
}

pub(crate) fn run(ds: &Dataset<f32, f32>) -> Result<(), DatasetParseError> {
    println!("=> Running knn classifier on flag dataset ...");

    run_distance(&ds)?;
    run_algorithm(&ds)?;
    run_weight(&ds)?;

    Ok(())
}
