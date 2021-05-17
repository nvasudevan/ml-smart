use crate::dataset;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::model_selection::train_test_split;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::metrics::{accuracy, mean_absolute_error};
use crate::dataset::DatasetParseError;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::naive_bayes::gaussian::GaussianNB;
use smartcore::neighbors::knn_classifier::KNNClassifier;
use smartcore::naive_bayes::categorical::CategoricalNB;
use smartcore::neighbors::knn_regressor::KNNRegressor;
use smartcore::naive_bayes::multinomial::MultinomialNB;
use crate::results::MLResult;
use crate::wine_quality::gaussian_regression;


fn knn_classify() -> Result<MLResult, DatasetParseError> {
    println!("=> Running KNN classifier on wine ...");
    let ds = dataset::wine::load_dataset(dataset::WINE_DATASET)?;
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );
    let knn_wine = KNNClassifier::fit(
        &nm_matrix, &ds.target, Default::default(),
    ).unwrap();

    //now try on test data
    let p = knn_wine.predict(&nm_matrix).unwrap();
    let res = MLResult::new( "kNN-classifier".to_string(),
                             accuracy(&ds.target, &p),
                             mean_absolute_error(&ds.target, &p)
    );

    Ok(res)
}


fn knn_regression() -> Result<MLResult, DatasetParseError> {
    println!("=> Running KNN regression on wine ...");
    let ds = dataset::wine::load_dataset(dataset::WINE_DATASET)?;
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );
    let model = KNNRegressor::fit(
        &nm_matrix, &ds.target, Default::default(),
    ).unwrap();

    //now try on test data
    let p = model.predict(&nm_matrix).unwrap();
    let res = MLResult::new( "kNN-regressor".to_string(),
                             accuracy(&ds.target, &p),
                             mean_absolute_error(&ds.target, &p)
    );

    Ok(res)
}

fn linear_regression() -> Result<MLResult, DatasetParseError> {
    println!("=> Running linear regression on Wine ...");
    let ds = dataset::wine::load_dataset(dataset::WINE_DATASET)?;
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
        true
    );
    let lnr_wine = LinearRegression::fit(
        &x_train, &y_train, Default::default(),
    ).unwrap();

    //now try on test data
    let p = lnr_wine.predict(&x_test).unwrap();
    let res = MLResult::new( "Linear Regression".to_string(),
                             accuracy(&y_test, &p),
                             mean_absolute_error(&y_test, &p)
    );

    Ok(res)
}

fn logistic_regression() -> Result<MLResult, DatasetParseError> {
    println!("=> Running logistic regression on Wine ...");
    let ds = dataset::wine::load_dataset(dataset::WINE_DATASET)?;
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
        true
    );
    let logr_wine = LogisticRegression::fit(
        &x_train, &y_train, Default::default(),
    ).unwrap();

    //now try on test data
    let p = logr_wine.predict(&x_test).unwrap();
    let res = MLResult::new( "Logistic Regression".to_string(),
                             accuracy(&y_test, &p),
                             mean_absolute_error(&y_test, &p)
    );

    Ok(res)
}

fn gaussianNB() -> Result<MLResult, DatasetParseError> {
    println!("=> Running gaussian regression on Wine ...");
    let ds = dataset::wine::load_dataset(dataset::WINE_DATASET)?;
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
        true
    );
    let guass_wine = GaussianNB::fit(
        &x_train, &y_train, Default::default(),
    ).unwrap();

    //now try on test data
    let p = guass_wine.predict(&x_test).unwrap();
    let res = MLResult::new( "Gaussian NB".to_string(),
                             accuracy(&y_test, &p),
                             mean_absolute_error(&y_test, &p)
    );

    Ok(res)
}

fn categoricalNB() -> Result<MLResult, DatasetParseError> {
    println!("=> Running categorical NB on Wine ...");
    let ds = dataset::wine::load_dataset(dataset::WINE_DATASET)?;
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
        true
    );
    let model = CategoricalNB::fit(
        &x_train, &y_train, Default::default(),
    ).unwrap();

    //now try on test data
    let p = model.predict(&x_test).unwrap();
    let res = MLResult::new( "Categorical NB".to_string(),
                             accuracy(&y_test, &p),
                             mean_absolute_error(&y_test, &p)
    );

    Ok(res)
}

fn multinomialNB() -> Result<MLResult, DatasetParseError> {
    println!("\n=> Running multinomial NB on Wine ...");
    let ds = dataset::wine::load_dataset(dataset::WINE_DATASET)?;
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
        true
    );
    let model = MultinomialNB::fit(
        &x_train, &y_train, Default::default(),
    ).unwrap();

    //now try on test data
    let p = model.predict(&x_test).unwrap();
    let res = MLResult::new( "Multinomial NB".to_string(),
                             accuracy(&y_test, &p),
                             mean_absolute_error(&y_test, &p)
    );

    Ok(res)
}

pub(crate) fn run() -> Result<Vec<MLResult>, DatasetParseError> {
    let mut results = Vec::<MLResult>::new();

    results.push(knn_classify()?);
    results.push(knn_regression()?);
    results.push(linear_regression()?);
    results.push(logistic_regression()?);
    results.push(gaussianNB()?);
    results.push(categoricalNB()?);
    results.push(multinomialNB()?);

    Ok(results)
}
