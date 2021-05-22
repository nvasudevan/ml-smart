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
use crate::dataset::DatasetParseError;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;

fn knn_classify(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running KNN classifier on Iris ...");
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );
    let knn = KNNClassifier::fit(
        &nm_matrix, &ds.target, Default::default(),
    )?;
    let p = knn.predict(&nm_matrix)?;

    let res = MLResult::new("kNN-classifier".to_string(),
                            accuracy(&ds.target, &p),
                            mean_absolute_error(&ds.target, &p),
    );

    Ok(res)
}

fn knn_regression(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running KNN regressor on Iris ...");
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );
    let knn = KNNRegressor::fit(
        &nm_matrix, &ds.target, Default::default(),
    )?;
    let p = knn.predict(&nm_matrix)?;
    let res = MLResult::new("kNN-regressor".to_string(),
                            accuracy(&ds.target, &p),
                            mean_absolute_error(&ds.target, &p),
    );

    Ok(res)
}

fn linear_regression(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running linear regression on Iris ...");
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );

    let model = LinearRegression::fit(
        &nm_matrix, &ds.target, Default::default(),
    )?;

    let p = model.predict(&nm_matrix)?;
    let res = MLResult::new("linear regression".to_string(),
                            accuracy(&ds.target, &p),
                            mean_absolute_error(&ds.target, &p),
    );

    Ok(res)
}

fn logistic_regression(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running logistic regression on Iris ...");
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );

    let lr_iris = LogisticRegression::fit(
        &nm_matrix, &ds.target, Default::default(),
    )?;

    let p = lr_iris.predict(&nm_matrix)?;
    let res = MLResult::new(
        "logistic regression".to_string(),
        accuracy(&ds.target, &p),
        mean_absolute_error(&ds.target, &p),
    );

    Ok(res)
}

fn gaussianNB(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running Gaussian on Iris ...");
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );

    let model = GaussianNB::fit(
        &nm_matrix, &ds.target, Default::default(),
    ).unwrap();

    let p = model.predict(&nm_matrix).unwrap();
    let res = MLResult::new(
        "gauss NB".to_string(),
        accuracy(&ds.target, &p),
        mean_absolute_error(&ds.target, &p),
    );

    Ok(res)
}

fn categoricalNB(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running categoricalNB on Iris ...");
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );

    let categorical_nb = CategoricalNB::fit(
        &nm_matrix, &ds.target, Default::default(),
    )?;

    let p = categorical_nb.predict(&nm_matrix)?;
    let res = MLResult::new(
        "categorical NB".to_string(),
        accuracy(&ds.target, &p),
        mean_absolute_error(&ds.target, &p),
    );

    Ok(res)
}

fn multinomialNB(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running multinomial on Iris ...");
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );

    let model = MultinomialNB::fit(
        &nm_matrix, &ds.target, Default::default(),
    )?;

    let p = model.predict(&nm_matrix)?;
    let res = MLResult::new(
        "multinomial NB".to_string(),
        accuracy(&ds.target, &p),
        mean_absolute_error(&ds.target, &p),
    );

    Ok(res)
}

fn tree_classifier(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running tree classifier on (full) iris dataset ...");
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );
    let model = DecisionTreeClassifier::fit(
        &nm_matrix, &ds.target, Default::default(),
    )?;

    let p = model.predict(&nm_matrix)?;
    // validate_predict(&ds, &p, &flag_recs);
    let res = MLResult::new("Decision Tree classifier (full)".to_string(),
                            accuracy(&ds.target, &p),
                            mean_absolute_error(&ds.target, &p),
    );

    Ok(res)
}

fn random_forest_classifier(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running forest classifier on iris dataset ...");
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );
    let model = RandomForestClassifier::fit(
        &nm_matrix, &ds.target, Default::default(),
    )?;

    let p = model.predict(&nm_matrix)?;
    // validate_predict(&ds, &p, flag_recs);
    let res = MLResult::new("Random forest classifier".to_string(),
                            accuracy(&ds.target, &p),
                            mean_absolute_error(&ds.target, &p),
    );

    Ok(res)
}

pub(crate) fn run() -> Result<Vec<MLResult>, DatasetParseError> {
    let ds = iris::load_dataset();
    let mut results = Vec::<MLResult>::new();

    results.push(knn_classify(&ds)?);
    results.push(knn_regression(&ds)?);
    results.push(linear_regression(&ds)?);
    results.push(logistic_regression(&ds)?);
    results.push(gaussianNB(&ds)?);
    results.push(categoricalNB(&ds)?);
    results.push(multinomialNB(&ds)?);
    results.push(tree_classifier(&ds)?);
    results.push(random_forest_classifier(&ds)?);

    Ok(results)
}