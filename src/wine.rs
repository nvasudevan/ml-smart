use crate::dataset;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::model_selection::train_test_split;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::metrics::{accuracy, mean_absolute_error};
use crate::dataset::DatasetParseError;

pub(crate) fn linear_regression() -> Result<(), DatasetParseError> {
    let ds = dataset::wine::load_dataset(dataset::WINE_DATASET)?;
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );
    let (x_train,
        x_test,
        y_train,
        y_test) = train_test_split(&nm_matrix, &ds.target, 0.4, true);
    println!("nXm[len={}]: {:?}", ds.num_samples, nm_matrix);
    println!("train: {:?}", x_train);
    let lnr_wine = LinearRegression::fit(
        &x_train, &y_train, Default::default(),
    ).unwrap();
    println!("lnr_wine: {:?}", lnr_wine);

    //now try on test data
    println!("\nx_test: {:?}", x_test);
    println!("y_test: {:?}", y_test);
    let p = lnr_wine.predict(&x_test).unwrap();
    println!("\np: {:?}", p);

    println!("accuracy: {}", accuracy(&y_test, &p));
    println!("mean abs error: {}", mean_absolute_error(&y_test, &p));

    Ok(())
}

