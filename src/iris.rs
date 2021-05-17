use smartcore::dataset::{iris, Dataset};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::neighbors::knn_classifier::KNNClassifier;
use smartcore::metrics::{accuracy, mean_absolute_error};
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::naive_bayes::gaussian::GaussianNB;
use smartcore::naive_bayes::categorical::CategoricalNB;
use smartcore::neighbors::knn_regressor::KNNRegressor;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::naive_bayes::multinomial::MultinomialNB;
use crate::results::MLResult;

fn knn_classify(results: &mut Vec<MLResult>) {
    println!("\n=> Running KNN classifier on Iris ...");
    let ds = iris::load_dataset();
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data
    );
    let ds_target = ds.target;

    let knn = KNNClassifier::fit(
        &nm_matrix, &ds_target, Default::default()).unwrap();
    let p= knn.predict(&nm_matrix).unwrap();
    // println!("\n=>p: {:?}", p);

    let res = MLResult::new( "kNN-classifier".to_string(),
                   accuracy(&ds_target, &p),
                   mean_absolute_error(&ds_target, &p)
    );
    results.push(res);
}

fn knn_regression(results: &mut Vec<MLResult>) {
    println!("\n=> Running KNN regressor on Iris ...");
    let ds = iris::load_dataset();
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data
    );
    let ds_target = ds.target;

    let knn = KNNRegressor::fit(
        &nm_matrix, &ds_target, Default::default()).unwrap();
    let p= knn.predict(&nm_matrix).unwrap();
    // println!("\n=>p: {:?}", p);

    let res = MLResult::new( "kNN-regressor".to_string(),
                   accuracy(&ds_target, &p),
                   mean_absolute_error(&ds_target, &p)
    );
    results.push(res);
}

fn linear_regression(results: &mut Vec<MLResult>) {
    println!("\n=> Running linear regression on Iris ...");
    let ds = iris::load_dataset();
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data
    );
    let ds_target = ds.target;

    let model = LinearRegression::fit(
        &nm_matrix, &ds_target, Default::default()
    ).unwrap();

    let p = model.predict(&nm_matrix).unwrap();
    // println!("p: {:?}", p);

    let res = MLResult::new( "linear regression".to_string(),
                   accuracy(&ds_target, &p),
                   mean_absolute_error(&ds_target, &p)
    );
    results.push(res);
}

fn logistic_regression(results: &mut Vec<MLResult>) {
    println!("\n=> Running logistic regression on Iris ...");
    let ds = iris::load_dataset();
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data
    );
    let ds_target = ds.target;

    let lr_iris = LogisticRegression::fit(
        &nm_matrix, &ds_target, Default::default()
    ).unwrap();

    let p = lr_iris.predict(&nm_matrix).unwrap();

    let res = MLResult::new(
        "logistic regression".to_string(),
        accuracy(&ds_target, &p),
        mean_absolute_error(&ds_target, &p)
    );
    results.push(res);
}

fn gaussianNB(results: &mut Vec<MLResult>) {
    println!("\n=> Running Gaussian on Iris ...");
    let ds = iris::load_dataset();
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data
    );
    let ds_target = ds.target;

    let model = GaussianNB::fit(
        &nm_matrix, &ds_target, Default::default()
    ).unwrap();

    let p = model.predict(&nm_matrix).unwrap();
    let res = MLResult::new(
        "gauss NB".to_string(),
        accuracy(&ds_target, &p),
        mean_absolute_error(&ds_target, &p)
    );

    results.push(res);
}

fn categoricalNB(results: &mut Vec<MLResult>) {
    println!("\n=> Running categoricalNB on Iris ...");
    let ds = iris::load_dataset();
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data
    );
    let ds_target = ds.target;

    let categorical_nb = CategoricalNB::fit(
        &nm_matrix, &ds_target, Default::default()
    ).unwrap();

    let p = categorical_nb.predict(&nm_matrix).unwrap();
    let res = MLResult::new(
        "categorical NB".to_string(),
        accuracy(&ds_target, &p),
        mean_absolute_error(&ds_target, &p)
    );
    results.push(res);
}

fn multinomialNB(results: &mut Vec<MLResult>) {
    println!("\n=> Running multinomial on Iris ...");
    let ds = iris::load_dataset();
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data
    );
    let ds_target = ds.target;

    let model = MultinomialNB::fit(
        &nm_matrix, &ds_target, Default::default()
    ).unwrap();

    let p = model.predict(&nm_matrix).unwrap();
    let res = MLResult::new(
        "multinomial NB".to_string(),
        accuracy(&ds_target, &p),
        mean_absolute_error(&ds_target, &p)
    );
    results.push(res);
}

pub(crate) fn run() -> Vec<MLResult> {
    let mut results = Vec::<MLResult>::new();

    knn_classify(&mut results);
    knn_regression(&mut results);
    linear_regression(&mut results);
    logistic_regression(&mut results);
    gaussianNB(&mut results);
    categoricalNB(&mut results);
    multinomialNB(&mut results);

    results
}