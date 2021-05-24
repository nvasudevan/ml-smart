mod random_forest;
mod knn_classifier;

use crate::results::MLResult;
use crate::dataset::{FLAG_DATASET, DatasetParseError, flag};
use smartcore::dataset::Dataset;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::neighbors::knn_classifier::KNNClassifier;
use smartcore::metrics::{accuracy, mean_absolute_error};
use smartcore::model_selection::train_test_split;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::naive_bayes::gaussian::GaussianNB;
use smartcore::naive_bayes::categorical::CategoricalNB;
use smartcore::naive_bayes::multinomial::MultinomialNB;
use smartcore::neighbors::knn_regressor::KNNRegressor;
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use crate::dataset::flag::Flag;

fn validate_predict(ds: &Dataset<f32, f32>, p: &Vec<f32>, flag_recs: &Vec<Flag>) {
    let mut n = 0;
    let mut unmatched_ctrys = Vec::<String>::new();
    for (i, v) in ds.target.iter().enumerate() {
        let p_val = p.get(i).unwrap();
        if v == p_val {
            n += 1;
        } else {
            let f = flag_recs.get(i)
                .expect("error retrieving flag from flags vector");
            unmatched_ctrys.push(f.country())
        }
    }
    println!("total match, n={}, acc %={}", n, (n as f32) / (ds.num_samples as f32));
    println!("unmatched: {}", unmatched_ctrys.join(", "));
}

// fn knn_classify(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
//     println!("=> Running kNN on flag dataset ...");
//     let nm_matrix = DenseMatrix::from_array(
//         ds.num_samples, ds.num_features, &ds.data,
//     );
//     let knn_wine = KNNClassifier::fit(
//         &nm_matrix, &ds.target, Default::default(),
//     )?;
//
//     let p = knn_wine.predict(&nm_matrix)?;
//     let res = MLResult::new("kNN-classifier".to_string(),
//                             accuracy(&ds.target, &p),
//                             mean_absolute_error(&ds.target, &p),
//     );
//
//     Ok(res)
// }
//

fn knn_regression(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running kNN regression on flag dataset ...");
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

fn tree_classifier_train(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running decision tree classifier on flag dataset ...");
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
    let logr = DecisionTreeClassifier::fit(
        &x_train, &y_train, Default::default(),
    )?;

    //now try on test data
    let p = logr.predict(&x_test)?;
    let res = MLResult::new("Decision Tree Classifer (train)".to_string(),
                            accuracy(&y_test, &p),
                            mean_absolute_error(&y_test, &p),
    );

    Ok(res)
}

fn tree_classifier(ds: &Dataset<f32, f32>, flag_recs: &Vec<Flag>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running tree classifier on (full) flag dataset ...");
    let nm_matrix = DenseMatrix::from_array(
        ds.num_samples, ds.num_features, &ds.data,
    );
    let model = DecisionTreeClassifier::fit(
        &nm_matrix, &ds.target, Default::default(),
    )?;

    let p = model.predict(&nm_matrix)?;
    validate_predict(&ds, &p, &flag_recs);
    let res = MLResult::new("Decision Tree classifier (full)".to_string(),
                            accuracy(&ds.target, &p),
                            mean_absolute_error(&ds.target, &p),
    );

    Ok(res)
}

fn logistic_regression(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running logistic regression on flag dataset ...");
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
    let logr = LogisticRegression::fit(
        &x_train, &y_train, Default::default(),
    )?;

    //now try on test data
    let p = logr.predict(&x_test)?;
    let res = MLResult::new("Logistic Regression".to_string(),
                            accuracy(&y_test, &p),
                            mean_absolute_error(&y_test, &p),
    );

    Ok(res)
}

fn gaussianNB(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running gaussian regression on flag ...");
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
    let gauss = GaussianNB::fit(
        &x_train, &y_train, Default::default(),
    )?;

    //now try on test data
    let p = gauss.predict(&x_test)?;
    let res = MLResult::new("Gaussian NB".to_string(),
                            accuracy(&y_test, &p),
                            mean_absolute_error(&y_test, &p),
    );

    Ok(res)
}

fn categoricalNB(ds: &Dataset<f32, f32>) -> Result<MLResult, DatasetParseError> {
    println!("=> Running categorical NB on flag dataset ...");
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
    println!("=> Running multinomial NB regression on flag dataset ...");
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

pub(crate) fn run_predict_religion() -> Result<Vec<MLResult>, DatasetParseError> {
    let (flag_recs, ds) = flag::load_dataset_tgt_religion(FLAG_DATASET)?;
    let mut results = Vec::<MLResult>::new();

    knn_classifier::run(&ds)?;
    // results.push(knn_regression(&ds)?);
    // // results.push(linear_regression(&ds)?);
    // results.push(tree_classifier_train(&ds)?);
    // results.push(tree_classifier(&ds, &flag_recs)?);
    random_forest::run(&ds)?;
    // results.push(logistic_regression(&ds)?);
    // // results.push(gaussianNB(&ds)?);
    // results.push(categoricalNB(&ds)?);
    // results.push(multinomialNB(&ds)?);

    Ok(results)
}

pub(crate) fn run_predict_language() -> Result<Vec<MLResult>, DatasetParseError> {
    let (flag_recs, ds) = flag::load_dataset_tgt_language(FLAG_DATASET)?;
    let mut results = Vec::<MLResult>::new();

    knn_classifier::run(&ds)?;
    // results.push(knn_regression(&ds)?);
    // results.push(linear_regression(&ds)?);
    // results.push(tree_classifier_train(&ds)?);
    // results.push(tree_classifier(&ds, &flag_recs)?);
    random_forest::run(&ds)?;
    // results.push(random_forest_classifier(&ds, &flag_recs)?);
    // results.push(logistic_regression(&ds)?);
    // // results.push(gaussianNB(&ds)?);
    // results.push(categoricalNB(&ds)?);
    // results.push(multinomialNB(&ds)?);

    Ok(results)
}

