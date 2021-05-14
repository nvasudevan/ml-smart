use crate::dataset;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::model_selection::train_test_split;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::metrics::{accuracy, mean_absolute_error};
use crate::dataset::DatasetParseError;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::naive_bayes::gaussian::GaussianNB;
use smartcore::neighbors::knn_classifier::KNNClassifier;


pub(crate) fn knn_classify() -> Result<(), DatasetParseError> {
    println!("\n=> Running KNN on wine ...");
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
    let knn_wine = KNNClassifier::fit(
        &x_train, &y_train, Default::default(),
    ).unwrap();

    //now try on test data
    let p = knn_wine.predict(&x_test).unwrap();
    println!("accuracy: {}", accuracy(&y_test, &p));
    println!("mean abs error: {}", mean_absolute_error(&y_test, &p));

    Ok(())
}

pub(crate) fn linear_regression() -> Result<(), DatasetParseError> {
    println!("\n=> Running linear regression on Wine ...");
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
    println!("accuracy: {}", accuracy(&y_test, &p));
    println!("mean abs error: {}", mean_absolute_error(&y_test, &p));

    Ok(())
}

pub(crate) fn logistic_regression() -> Result<(), DatasetParseError> {
    println!("\n=> Running logistic regression on Wine ...");
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
    println!("accuracy: {}", accuracy(&y_test, &p));
    println!("mean abs error: {}", mean_absolute_error(&y_test, &p));

    Ok(())
}

pub(crate) fn gaussian_regression() -> Result<(), DatasetParseError> {
    println!("\n=> Running gaussian regression on Wine ...");
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
    println!("accuracy: {}", accuracy(&y_test, &p));
    println!("mean abs error: {}", mean_absolute_error(&y_test, &p));

    Ok(())
}

