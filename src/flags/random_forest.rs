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
    let logr = RandomForestClassifier::fit(
        &x_train, &y_train, params,
    )?;

    //now try on test data
    let p = logr.predict(&x_test)?;
    let res = MLResult::new("Random forest Classifer (train)".to_string(),
                            accuracy(&y_test, &p),
                            mean_absolute_error(&y_test, &p),
    );

    Ok(res)
}

fn tweak_no_trees(ds: &Dataset<f32, f32>, criterion: &SplitCriterion)
                  -> Result<(f32, RandomForestClassifierParameters), DatasetParseError> {
    let mut no_changes = 0;
    let mut n_iter = 0;
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
        // println!("{}:{}.2 [{}]", n_trees, res_acc, no_changes);
        if res_acc >= acc {
            acc = res_acc;
            n_trees += 1;
            no_changes = 0;
        } else {
            no_changes += 1;
        }

        n_iter += 1;
        if no_changes >= crate::MAX_NO_CHANGES {
            return Ok(
                (
                    acc,
                    RandomForestClassifierParameters {
                        criterion: criterion.clone(),
                        max_depth: None,
                        min_samples_leaf: 1,
                        min_samples_split: 2,
                        n_trees,
                        m: Option::None,
                    }
                )
            );
        }
    }
}

pub(crate) fn run(ds: &Dataset<f32, f32>) -> Result<(), DatasetParseError> {
    println!("=> Running random forest classifier on flag dataset ...");
    let splits = [
        SplitCriterion::Gini,
        SplitCriterion::Entropy,
        SplitCriterion::ClassificationError
    ];

    for crit in splits.iter() {
        let (acc, params) = tweak_no_trees(
            &ds, crit
        )?;
        println!("best accuracy: {} (params: {:?})", acc, params);

    }

    Ok(())
}
