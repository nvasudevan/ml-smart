use smartcore::dataset::Dataset;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::model_selection::train_test_split;
use smartcore::tree::decision_tree_classifier::{
    DecisionTreeClassifier,
    SplitCriterion,
    DecisionTreeClassifierParameters
};
use smartcore::metrics::{accuracy, mean_absolute_error};

use crate::dataset::DatasetParseError;
use crate::dataset::flag::Flag;
use crate::flags::validate_predict;
use crate::results::MLResult;

fn params_as_str(params: &DecisionTreeClassifierParameters) -> String {
    match params.criterion {
        SplitCriterion::Entropy => {
            return format!("Decision Tree Classifier [{}]", "Entropy");
        },
        SplitCriterion::ClassificationError => {
            return format!("Decision Tree Classifier [{}]", "ClassificationError");
        },
        _ => {}
    }

    format!("Decision Tree Classifier [{}]", "Gini")
}

fn train_and_test(ds: &Dataset<f32, f32>, criterion: &SplitCriterion)
    -> Result<MLResult, DatasetParseError> {
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
    let mut params = DecisionTreeClassifierParameters::default()
        .with_criterion(criterion.clone());
    let params_s = params_as_str(&params);

    let logr = DecisionTreeClassifier::fit(
        &x_train, &y_train, params,
    )?;

    //now try on test data
    let p = logr.predict(&x_test)?;
    let res = MLResult::new(params_s,
                            accuracy(&y_test, &p),
                            mean_absolute_error(&y_test, &p),
    );

    Ok(res)
}

pub(crate) fn run(ds: &Dataset<f32, f32>) -> Result<Vec<MLResult>, DatasetParseError> {
    let mut results = Vec::<MLResult>::new();
    let splits = [
        SplitCriterion::Gini,
        SplitCriterion::Entropy,
        SplitCriterion::ClassificationError
    ];
    for crit in splits.iter() {
       let res = train_and_test(ds, crit)?;
        results.push(res);
    }

    Ok(results)
}

// fn full_run(ds: &Dataset<f32, f32>, flag_recs: &Vec<Flag>) -> Result<MLResult, DatasetParseError> {
//     println!("=> Running tree classifier on (full) flag dataset ...");
//     let nm_matrix = DenseMatrix::from_array(
//         ds.num_samples, ds.num_features, &ds.data,
//     );
//     let model = DecisionTreeClassifier::fit(
//         &nm_matrix, &ds.target, Default::default(),
//     )?;
//
//     let p = model.predict(&nm_matrix)?;
//     validate_predict(&ds, &p, &flag_recs);
//     let res = MLResult::new("Decision Tree classifier (full)".to_string(),
//                             accuracy(&ds.target, &p),
//                             mean_absolute_error(&ds.target, &p),
//     );
//
//     Ok(res)
// }

