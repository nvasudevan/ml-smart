use smartcore::dataset::{iris, Dataset};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::neighbors::knn_classifier::KNNClassifier;
use smartcore::metrics::accuracy;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::naive_bayes::gaussian::GaussianNB;

pub(crate) fn knn_classify() {
    let ds = iris::load_dataset();
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data
    );
    let ds_target = ds.target;

    let knn = KNNClassifier::fit(
        &nm_matrix, &ds_target, Default::default()).unwrap();
    let p= knn.predict(&nm_matrix).unwrap();
    println!("\n=>p: {:?}", p);

    let acc = accuracy(&ds_target, &p);
    println!("accuracy: {}", acc);
}

pub(crate) fn regression() {
    let ds = iris::load_dataset();
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data
    );
    let ds_target = ds.target;

    let lr_iris = LogisticRegression::fit(
        &nm_matrix, &ds_target, Default::default()
    ).unwrap();
    println!("lr: {:?}", lr_iris);

    let p = lr_iris.predict(&nm_matrix).unwrap();
    println!("p: {:?}", p);

    let acc = accuracy(&ds_target, &p);
    println!("accuracy: {}", acc);
}

pub(crate) fn guassian() {
    let ds = iris::load_dataset();
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data
    );
    let ds_target = ds.target;

    let guassian_nb = GaussianNB::fit(
        &nm_matrix, &ds_target, Default::default()
    ).unwrap();
    println!("guassian_nb: {:?}", guassian_nb);

    let p = guassian_nb.predict(&nm_matrix).unwrap();
    println!("guass nb acc: {}", accuracy(&ds_target, &p));
}