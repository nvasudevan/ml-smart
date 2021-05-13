use crate::dataset;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::model_selection::train_test_split;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::metrics::{accuracy, mean_absolute_error};
use crate::dataset::DatasetParseError;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::naive_bayes::gaussian::GaussianNB;


pub(crate) fn linear_regression() -> Result<(), DatasetParseError> {
    println!("\n=> Running linear regression on Wine ...");
    let ds = dataset::wine::load_dataset(dataset::WINE_DATASET)?;
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );
    let (x_train,
        x_test,
        y_train,
        y_test) = train_test_split(&nm_matrix, &ds.target, 0.8, true);
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
        y_test) = train_test_split(&nm_matrix, &ds.target, 0.8, true);
    let logr_wine = LogisticRegression::fit(
        &x_train, &y_train, Default::default(),
    ).unwrap();
    // println!("logr_wine: {:?}", logr_wine);

    //now try on test data
    let p = logr_wine.predict(&x_test).unwrap();
    println!("accuracy: {}", accuracy(&y_test, &p));
    println!("mean abs error: {}", mean_absolute_error(&y_test, &p));

    Ok(())
}

pub(crate) fn guassian_regression() -> Result<(), DatasetParseError> {
    println!("\n=> Running guassian regression on Wine ...");
    let ds = dataset::wine::load_dataset(dataset::WINE_DATASET)?;
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );
    let (x_train,
        x_test,
        y_train,
        y_test) = train_test_split(&nm_matrix, &ds.target, 0.8, true);
    let guass_wine = GaussianNB::fit(
        &x_train, &y_train, Default::default(),
    ).unwrap();

    //now try on test data
    let p = guass_wine.predict(&x_test).unwrap();
    println!("accuracy: {}", accuracy(&y_test, &p));
    println!("mean abs error: {}", mean_absolute_error(&y_test, &p));

    Ok(())
}

