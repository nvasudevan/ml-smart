use smartcore::dataset::boston;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::metrics::{accuracy, mean_absolute_error};
use smartcore::model_selection::train_test_split;

pub(crate) fn linear_regression() {
    println!("\n=> Running linear regression on Boston ...");
    let ds = boston::load_dataset();
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );
    let (x_train,
         x_test,
         y_train,
         y_test) = train_test_split(&nm_matrix, &ds.target, 0.8, true);
    println!("nXm[len={}]: {:?}", ds.num_samples, nm_matrix);
    let lnr_boston = LinearRegression::fit(
        &x_train, &y_train, Default::default(),
    ).unwrap();
    println!("lnr_boston: {:?}", lnr_boston);

    //now try on test data
    let p = lnr_boston.predict(&x_test).unwrap();
    println!("accuracy: {}", accuracy(&y_test, &p));
    println!("mean abs error: {}", mean_absolute_error(&y_test, &p));
}

pub(crate) fn logistic_regression() {
    println!("\n=> Running logistic regression on Boston ...");
    let ds = boston::load_dataset();
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );
    let (x_train,
        x_test,
        y_train,
        y_test) = train_test_split(&nm_matrix, &ds.target, 0.6, true);

    let lr_boston = LogisticRegression::fit(
        &x_train, &y_train, Default::default(),
    ).unwrap();

    let p = lr_boston.predict(&x_test).unwrap();
    println!("accuracy: {}", accuracy(&y_test, &p));
    println!("mean abs error: {}", mean_absolute_error(&y_test, &p));
}