use crate::dataset;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::model_selection::train_test_split;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::metrics::{accuracy, mean_absolute_error};
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::naive_bayes::gaussian::GaussianNB;
use smartcore::neighbors::knn_classifier::KNNClassifier;
use smartcore::naive_bayes::categorical::CategoricalNB;
use smartcore::neighbors::knn_regressor::KNNRegressor;
use smartcore::naive_bayes::multinomial::MultinomialNB;
use crate::results::MLResult;
use smartcore::dataset::Dataset;
use crate::dataset::DatasetParseError;
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;


fn knn_classify(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running KNN classifier on wine ...");
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );
    let knn_wine = KNNClassifier::fit(
        &nm_matrix, &ds.target, Default::default(),
    )?;

    //now try on test data
    let p = knn_wine.predict(&nm_matrix)?;
    let res = MLResult::new("kNN-classifier".to_string(),
                            accuracy(&ds.target, &p),
                            mean_absolute_error(&ds.target, &p),
    );

    Ok(res)
}


fn knn_regression(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running KNN regression on wine ...");
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );
    let model = KNNRegressor::fit(
        &nm_matrix, &ds.target, Default::default(),
    )?;

    //now try on test data
    let p = model.predict(&nm_matrix)?;
    let res = MLResult::new("kNN-regressor".to_string(),
                            accuracy(&ds.target, &p),
                            mean_absolute_error(&ds.target, &p),
    );

    Ok(res)
}

fn linear_regression(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running linear regression on Wine ...");
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
    let lnr_wine = LinearRegression::fit(
        &x_train, &y_train, Default::default(),
    )?;

    //now try on test data
    let p = lnr_wine.predict(&x_test)?;
    let res = MLResult::new("Linear Regression".to_string(),
                            accuracy(&y_test, &p),
                            mean_absolute_error(&y_test, &p),
    );

    Ok(res)
}

fn logistic_regression(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running logistic regression on Wine ...");
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
    let logr_wine = LogisticRegression::fit(
        &x_train, &y_train, Default::default(),
    )?;

    //now try on test data
    let p = logr_wine.predict(&x_test)?;
    let res = MLResult::new("Logistic Regression".to_string(),
                            accuracy(&y_test, &p),
                            mean_absolute_error(&y_test, &p),
    );

    Ok(res)
}

fn gaussianNB(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running gaussian regression on Wine ...");
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
    let guass_wine = GaussianNB::fit(
        &x_train, &y_train, Default::default(),
    )?;

    //now try on test data
    let p = guass_wine.predict(&x_test)?;
    let res = MLResult::new("Gaussian NB".to_string(),
                            accuracy(&y_test, &p),
                            mean_absolute_error(&y_test, &p),
    );

    Ok(res)
}

fn categoricalNB(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running categorical NB on Wine ...");
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
    )?;

    //now try on test data
    let p = model.predict(&x_test)?;
    let res = MLResult::new("Categorical NB".to_string(),
                            accuracy(&y_test, &p),
                            mean_absolute_error(&y_test, &p),
    );

    Ok(res)
}

fn multinomialNB(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running multinomial NB on Wine ...");
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
    let model = MultinomialNB::fit(
        &x_train, &y_train, Default::default(),
    )?;

    //now try on test data
    let p = model.predict(&x_test)?;
    let res = MLResult::new("Multinomial NB".to_string(),
                            accuracy(&y_test, &p),
                            mean_absolute_error(&y_test, &p),
    );

    Ok(res)
}

fn tree_classifier(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running tree classifier on (full) wine class dataset ...");
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
    println!("=> Running forest classifier on wine class dataset ...");
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
    let ds = dataset::wine::load_dataset(dataset::WINE_DATASET)?;
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
