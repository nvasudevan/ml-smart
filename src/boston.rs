use smartcore::dataset::Dataset;
use crate::dataset;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::metrics::{accuracy, mean_absolute_error};
use smartcore::model_selection::train_test_split;
use smartcore::naive_bayes::gaussian::GaussianNB;
use smartcore::neighbors::knn_classifier::KNNClassifier;
use smartcore::neighbors::knn_regressor::KNNRegressor;
use smartcore::naive_bayes::categorical::CategoricalNB;
use crate::results::MLResult;
use smartcore::naive_bayes::multinomial::MultinomialNB;
use crate::dataset::DatasetParseError;

fn knn_classify(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running KNN classifier on Boston ...");
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );
    let knn_boston = KNNClassifier::fit(
        &nm_matrix, &ds.target, Default::default(),
    )?;

    //now try on test data
    let p = knn_boston.predict(&nm_matrix)?;
    let res = MLResult::new("kNN-classifier".to_string(),
                            accuracy(&ds.target, &p),
                            mean_absolute_error(&ds.target, &p),
    );

    Ok(res)
}

fn knn_regression(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running KNN regression on Boston ...");
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );
    let knn_boston = KNNRegressor::fit(
        &nm_matrix, &ds.target, Default::default(),
    ).unwrap();

    //now try on test data
    let p = knn_boston.predict(&nm_matrix)?;
    let res = MLResult::new("kNN-regressor".to_string(),
                            accuracy(&ds.target, &p),
                            mean_absolute_error(&ds.target, &p),
    );

    Ok(res)
}

fn linear_regression(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running linear regression on Boston ...");
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
    let lnr_boston = LinearRegression::fit(
        &x_train, &y_train, Default::default(),
    )?;

    //now try on test data
    let p = lnr_boston.predict(&x_test)?;
    let res = MLResult::new("Linear Regression".to_string(),
                            accuracy(&y_test, &p),
                            mean_absolute_error(&y_test, &p),
    );

    Ok(res)
}

fn logistic_regression(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running logistic regression on Boston ...");
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

    let lr_boston = LogisticRegression::fit(
        &x_train, &y_train, Default::default(),
    )?;

    let p = lr_boston.predict(&x_test)?;
    let res = MLResult::new("Logistic Regression".to_string(),
                            accuracy(&y_test, &p),
                            mean_absolute_error(&y_test, &p),
    );

    Ok(res)
}

fn gaussianNB(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running gaussian NB on Boston ...");
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

    let model = GaussianNB::fit(
        &x_train, &y_train, Default::default(),
    )?;

    let p = model.predict(&x_test)?;
    let res = MLResult::new("Gaussian NB".to_string(),
                            accuracy(&y_test, &p),
                            mean_absolute_error(&y_test, &p),
    );

    Ok(res)
}

fn categoricalNB(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running categorical NB on Boston ...");
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

    let model = CategoricalNB::fit(
        &x_train, &y_train, Default::default(),
    )?;

    let p = model.predict(&x_test)?;
    let res = MLResult::new("Categorical NB".to_string(),
                            accuracy(&y_test, &p),
                            mean_absolute_error(&y_test, &p),
    );

    Ok(res)
}

fn multinomialNB(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running categorical NB on Boston ...");
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

    let model = MultinomialNB::fit(
        &x_train, &y_train, Default::default(),
    )?;

    let p = model.predict(&x_test)?;
    let res = MLResult::new("Multinomial NB".to_string(),
                            accuracy(&y_test, &p),
                            mean_absolute_error(&y_test, &p),
    );

    Ok(res)
}

pub(crate) fn run() -> Result<Vec<MLResult>, DatasetParseError> {
    let ds = dataset::wine::load_dataset(dataset::WINE_DATASET)?;
    let mut results = Vec::<MLResult>::new();

    results.push(knn_classify(&ds)?);
    results.push(knn_regression(&ds)?);
    results.push(linear_regression(&ds)?);
    results.push(logistic_regression(&ds)?);
    // results.push(gaussianNB(&ds)?);
    results.push(categoricalNB(&ds)?);
    results.push(multinomialNB(&ds)?);

    Ok(results)
}
