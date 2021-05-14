use smartcore::dataset::boston;
use crate::dataset;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::metrics::{accuracy, mean_absolute_error};
use smartcore::model_selection::train_test_split;
use smartcore::naive_bayes::gaussian::GaussianNB;
use smartcore::neighbors::knn_classifier::KNNClassifier;

pub(crate) fn knn_classify() {
    println!("\n=> Running KNN on Boston ...");
    let ds = boston::load_dataset();
    println!("ds: samples={}, features={}, target={}",
             ds.num_samples, ds.num_features, ds.target_names.join(", "));
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
    let knn_boston = KNNClassifier::fit(
        &x_train, &y_train, Default::default(),
    ).unwrap();

    //now try on test data
    let p = knn_boston.predict(&x_test).unwrap();
    println!("accuracy: {}", accuracy(&y_test, &p));
    println!("mean abs error: {}", mean_absolute_error(&y_test, &p));
}

pub(crate) fn linear_regression() {
    println!("\n=> Running linear regression on Boston ...");
    let ds = boston::load_dataset();
    println!("ds: samples={}, features={}, target={}",
             ds.num_samples, ds.num_features, ds.target_names.join(", "));
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
    ).unwrap();

    //now try on test data
    let p = lnr_boston.predict(&x_test).unwrap();
    println!("accuracy: {}", accuracy(&y_test, &p));
    println!("mean abs error: {}", mean_absolute_error(&y_test, &p));
}

pub(crate) fn logistic_regression() {
    println!("\n=> Running logistic regression on Boston ...");
    let ds = boston::load_dataset();
    println!("ds: samples={}, features={}, target={}",
             ds.num_samples, ds.num_features, ds.target_names.join(", "));
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
    ).unwrap();

    let p = lr_boston.predict(&x_test).unwrap();
    println!("accuracy: {}", accuracy(&y_test, &p));
    println!("mean abs error: {}", mean_absolute_error(&y_test, &p));
}