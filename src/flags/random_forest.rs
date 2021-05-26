use smartcore::dataset::Dataset;
use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier,
    RandomForestClassifierParameters,
};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::metrics::{accuracy, mean_absolute_error};
use smartcore::model_selection::train_test_split;
use smartcore::tree::decision_tree_classifier::SplitCriterion;

use crate::dataset::DatasetParseError;
use crate::dataset::flag::Flag;
use crate::flags::validate_predict;
use crate::results::MLResult;

fn random_forest_params_as_str(params: &RandomForestClassifierParameters) -> String {
    match params.criterion {
        SplitCriterion::Entropy => {
            return format!("RandomForest Classifier [{}] n_trees={}", "Entropy", params.n_trees);
        },
        SplitCriterion::ClassificationError => {
            return format!("RandomForest Classifier [{}] n_trees={}", "ClassificationError", params.n_trees);
        },
        _ => {}
    }

    format!("RandomForest Classifier [{}] n_trees={}", "Gini", params.n_trees)
}

fn train_and_test(ds: &Dataset<f32, f32>, params: RandomForestClassifierParameters)
    -> Result<MLResult, DatasetParseError> {
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
    let params_s = random_forest_params_as_str(&params);
    let logr = RandomForestClassifier::fit(
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

fn tweak_no_trees(ds: &Dataset<f32, f32>, criterion: &SplitCriterion)
                  -> Result<MLResult, DatasetParseError> {
    let mut no_changes = 0;
    let mut acc = 0.0;
    let mut n_trees = 10;
    loop {
        let params = RandomForestClassifierParameters {
            criterion: criterion.clone(),
            max_depth: None,
            min_samples_leaf: 1,
            min_samples_split: 2,
            n_trees,
            m: Option::None,
        };
        let res = train_and_test(&ds, params)?;
        let res_acc = res.acc();
        if res_acc >= acc {
            acc = res_acc;
            n_trees += 1;
            no_changes = 0;
        } else {
            no_changes += 1;
        }

        if no_changes >= crate::MAX_NO_CHANGES {
            let opt_params = RandomForestClassifierParameters {
                criterion: criterion.clone(),
                max_depth: None,
                min_samples_leaf: 1,
                min_samples_split: 2,
                n_trees,
                m: Option::None,
            };
            let opt_res =  MLResult::new(res.name(), res_acc, res.mae());
            return Ok(opt_res);
        }
    }
}

pub(crate) fn run(ds: &Dataset<f32, f32>) -> Result<Vec<MLResult>, DatasetParseError> {
    println!("\n=> Running random forest classifier on flag dataset ...");
    let splits = [
        SplitCriterion::Gini,
        SplitCriterion::Entropy,
        SplitCriterion::ClassificationError
    ];

    let mut results = Vec::<MLResult>::new();
    for crit in splits.iter() {
        results.push(tweak_no_trees( &ds, crit)?);
    }

    Ok(results)
}
