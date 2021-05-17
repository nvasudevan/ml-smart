use smartcore::dataset::boston;
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

pub(crate) fn knn_classify(results: &mut Vec<MLResult>) {
    println!("=> Running KNN classifier on Boston ...");
    let ds = boston::load_dataset();
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );
    let knn_boston = KNNClassifier::fit(
        &nm_matrix, &ds.target, Default::default(),
    ).unwrap();

    //now try on test data
    let p = knn_boston.predict(&nm_matrix).unwrap();

    let res = MLResult::new( "kNN-classifier".to_string(),
                             accuracy(&ds.target, &p),
                             mean_absolute_error(&ds.target, &p)
    );
    results.push(res);
}

pub(crate) fn knn_regression(results: &mut Vec<MLResult>) {
    println!("=> Running KNN regression on Boston ...");
    let ds = boston::load_dataset();
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );
    let knn_boston = KNNRegressor::fit(
        &nm_matrix, &ds.target, Default::default(),
    ).unwrap();

    //now try on test data
    let p = knn_boston.predict(&nm_matrix).unwrap();
    let res = MLResult::new( "kNN-regressor".to_string(),
                             accuracy(&ds.target, &p),
                             mean_absolute_error(&ds.target, &p)
    );
    results.push(res);
}

pub(crate) fn linear_regression(results: &mut Vec<MLResult>) {
    println!("=> Running linear regression on Boston ...");
    let ds = boston::load_dataset();
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
    let res = MLResult::new( "Linear Regression".to_string(),
                             accuracy(&y_test, &p),
                             mean_absolute_error(&y_test, &p)
    );
    results.push(res);
}

pub(crate) fn logistic_regression(results: &mut Vec<MLResult>) {
    println!("=> Running logistic regression on Boston ...");
    let ds = boston::load_dataset();
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
    let res = MLResult::new( "Logistic Regression".to_string(),
                             accuracy(&y_test, &p),
                             mean_absolute_error(&y_test, &p)
    );
    results.push(res);
}

pub(crate) fn gaussianNB(results: &mut Vec<MLResult>) {
    println!("=> Running gaussian NB on Boston ...");
    let ds = boston::load_dataset();
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
    ).unwrap();

    let p = model.predict(&x_test).unwrap();
    let res = MLResult::new( "Gaussian NB".to_string(),
                             accuracy(&y_test, &p),
                             mean_absolute_error(&y_test, &p)
    );
    results.push(res);
}

pub(crate) fn categoricalNB(results: &mut Vec<MLResult>) {
    println!("=> Running categorical NB on Boston ...");
    let ds = boston::load_dataset();
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
    ).unwrap();

    let p = model.predict(&x_test).unwrap();
    let res = MLResult::new( "Categorical NB".to_string(),
                             accuracy(&y_test, &p),
                             mean_absolute_error(&y_test, &p)
    );
    results.push(res);
}

pub(crate) fn run() -> Vec<MLResult> {
    let mut results = Vec::<MLResult>::new();

    knn_classify(&mut results);
    knn_regression(&mut results);
    linear_regression(&mut results);
    logistic_regression(&mut results);
    // gaussianNB(&mut results);
    categoricalNB(&mut results);
    // multinomialNB(&mut results);

    results
}
