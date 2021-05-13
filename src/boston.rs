use smartcore::dataset::boston;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::metrics::{accuracy, mean_absolute_error};
use smartcore::model_selection::train_test_split;

pub(crate) fn linear_regression() {
    let ds = boston::load_dataset();
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );
    let (x_train,
         x_test,
         y_train,
         y_test) = train_test_split(&nm_matrix, &ds.target, 0.6, true);
    println!("nXm[len={}]: {:?}", ds.num_samples, nm_matrix);
    println!("train: {:?}", x_train);
    let lnr_boston = LinearRegression::fit(
        &x_train, &y_train, Default::default(),
    ).unwrap();
    println!("lnr_boston: {:?}", lnr_boston);

    //now try on test data
    println!("\nx_test: {:?}", x_test);
    println!("y_test: {:?}", y_test);
    let p = lnr_boston.predict(&x_test).unwrap();
    println!("\np: {:?}", p);

    println!("accuracy: {}", accuracy(&y_test, &p));
    println!("mean abs error: {}", mean_absolute_error(&y_test, &p));
}

pub(crate) fn logistic_regression() {
    let ds = boston::load_dataset();
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );
    println!("nXm[len={}]: {:?}", ds.num_samples, nm_matrix);

    let lr_boston = LogisticRegression::fit(
        &nm_matrix, &ds.target, Default::default(),
    ).unwrap();
    println!("lr_boston: {:?}", lr_boston);
}