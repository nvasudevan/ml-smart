use smartcore::neighbors::knn_classifier::KNNClassifier;
use smartcore::model_selection::train_test_split;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use crate::dataset::{DatasetParseError, wine_quality};
use smartcore::metrics::{accuracy, mean_absolute_error};
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::naive_bayes::gaussian::GaussianNB;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::neighbors::knn_regressor::KNNRegressor;
use smartcore::naive_bayes::categorical::CategoricalNB;
use smartcore::naive_bayes::multinomial::MultinomialNB;
use crate::results::MLResult;
use smartcore::dataset::Dataset;

fn knn_classify(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running kNN on wine quality ...");
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );
    let knn_wine = KNNClassifier::fit(
        &nm_matrix, &ds.target, Default::default(),
    )?;

    let p = knn_wine.predict(&nm_matrix)?;
    let res = MLResult::new("kNN-classifier".to_string(),
                            accuracy(&ds.target, &p),
                            mean_absolute_error(&ds.target, &p),
    );

    Ok(res)
}

fn knn_regression(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running kNN regression on wine quality ...");
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );
    let model = KNNRegressor::fit(
        &nm_matrix, &ds.target, Default::default(),
    )?;

    //now try on test data
    let p = model.predict(&nm_matrix)?;
    let res = MLResult::new("kNN-regressor".to_string(),
                            accuracy(&ds.target, &p),
                            mean_absolute_error(&ds.target, &p),
    );

    Ok(res)
}

fn logistic_regression(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running logistic regression on wine quality...");
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
    let logr = LogisticRegression::fit(
        &x_train, &y_train, Default::default(),
    )?;

    //now try on test data
    let p = logr.predict(&x_test)?;
    let res = MLResult::new("Logistic Regression".to_string(),
                            accuracy(&y_test, &p),
                            mean_absolute_error(&y_test, &p),
    );

    Ok(res)
}

fn linear_regression(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running linear regression on wine quality...");
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
    let lr = LinearRegression::fit(
        &x_train, &y_train, Default::default(),
    )?;

    //now try on test data
    let p = lr.predict(&x_test)?;
    let res = MLResult::new("Linear Regression".to_string(),
                            accuracy(&y_test, &p),
                            mean_absolute_error(&y_test, &p),
    );

    Ok(res)
}

fn gaussianNB(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running gaussian regression on wine quality...");
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
    let gauss = GaussianNB::fit(
        &x_train, &y_train, Default::default(),
    )?;

    //now try on test data
    let p = gauss.predict(&x_test)?;
    let res = MLResult::new("Gaussian NB".to_string(),
                            accuracy(&y_test, &p),
                            mean_absolute_error(&y_test, &p),
    );

    Ok(res)
}

fn categoricalNB(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running categorical NB on wine quality...");
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

    //now try on test data
    let p = model.predict(&x_test)?;
    let res = MLResult::new("Categorical NB".to_string(),
                            accuracy(&y_test, &p),
                            mean_absolute_error(&y_test, &p),
    );

    Ok(res)
}

fn multinomialNB(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running multinomial NB regression on wine quality...");
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

    //now try on test data
    let p = model.predict(&x_test)?;
    let res = MLResult::new("Multinomial NB".to_string(),
                            accuracy(&y_test, &p),
                            mean_absolute_error(&y_test, &p),
    );

    Ok(res)
}

pub(crate) fn run_red() -> Result<Vec<MLResult>, DatasetParseError> {
    let ds = wine_quality::load_red_dataset()?;
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

pub(crate) fn run_white() -> Result<Vec<MLResult>, DatasetParseError> {
    let ds = wine_quality::load_white_dataset()?;
    let mut results = Vec::<MLResult>::new();

    results.push(knn_classify(&ds)?);
    results.push(knn_regression(&ds)?);
    results.push(linear_regression(&ds)?);
    results.push(logistic_regression(&ds)?);
    results.push(gaussianNB(&ds)?);
    results.push(categoricalNB(&ds)?);
    results.push(multinomialNB(&ds)?);

    Ok(results)
}
