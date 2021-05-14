use crate::dataset;
use smartcore::neighbors::knn_classifier::KNNClassifier;
use smartcore::model_selection::train_test_split;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use crate::dataset::DatasetParseError;
use smartcore::metrics::{accuracy, mean_absolute_error};
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::naive_bayes::gaussian::GaussianNB;
use smartcore::linear::linear_regression::LinearRegression;

pub(crate) fn knn_classify() -> Result<(), DatasetParseError> {
    println!("\n=> Running kNN on wine quality ...");
    let ds = dataset::wine_quality::load_red_dataset()?;
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
    let knn_wine = KNNClassifier::fit(
        &x_train, &y_train, Default::default(),
    ).unwrap();

    //now try on test data
    let p = knn_wine.predict(&x_test).unwrap();
    println!("accuracy: {}", accuracy(&y_test, &p));
    println!("mean abs error: {}", mean_absolute_error(&y_test, &p));

    Ok(())
}

pub(crate) fn logistic_regression() -> Result<(), DatasetParseError> {
    println!("\n=> Running logistic regression on wine quality...");
    let ds = dataset::wine_quality::load_red_dataset()?;
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
    ).unwrap();

    //now try on test data
    let p = logr.predict(&x_test).unwrap();
    println!("accuracy: {}", accuracy(&y_test, &p));
    println!("mean abs error: {}", mean_absolute_error(&y_test, &p));

    Ok(())
}

pub(crate) fn gaussian_regression() -> Result<(), DatasetParseError> {
    println!("\n=> Running gaussian regression on wine quality...");
    let ds = dataset::wine_quality::load_red_dataset()?;
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
    ).unwrap();

    //now try on test data
    let p = gauss.predict(&x_test).expect("Gaussian predict failed!");
    println!("accuracy: {}", accuracy(&y_test, &p));
    println!("mean abs error: {}", mean_absolute_error(&y_test, &p));

    Ok(())
}

pub(crate) fn linear_regression() -> Result<(), DatasetParseError> {
    println!("\n=> Running linear regression on wine quality...");
    let ds = dataset::wine_quality::load_red_dataset()?;
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
    ).unwrap();

    //now try on test data
    let p = lr.predict(&x_test).unwrap();
    println!("accuracy: {}", accuracy(&y_test, &p));
    println!("mean abs error: {}", mean_absolute_error(&y_test, &p));

    Ok(())
}