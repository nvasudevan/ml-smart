use std::{
    io,
    fmt,
    num::ParseFloatError
};
use std::fmt::Formatter;
use smartcore::error::Failed;
use std::num::ParseIntError;
use crate::results::MLResult;
use smartcore::model_selection::train_test_split;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::dataset::Dataset;

pub(crate) mod wine;
pub(crate) mod wine_quality;
pub(crate) mod flag;

// path of various datasets
pub(crate) const FLAG_DATASET: &str = "./datasets/flags/flag.data";
pub(crate) const WINE_DATASET: &str = "./datasets/wine/class/wine.data";
pub(crate) const WINE_RED_QUALITY_DATASET: &str = "./datasets/wine/quality/winequality-red.csv";
pub(crate) const WINE_WHITE_QUALITY_DATASET: &str = "./datasets/wine/quality/winequality-white.csv";

pub(crate) struct TrainTestDataset<'a> {
    pub(crate) ds: &'a Dataset<f32, f32>,
    pub(crate) x_train: DenseMatrix<f32>,
    pub(crate) y_train: Vec<f32>,
    pub(crate) x_test: DenseMatrix<f32>,
    pub(crate) y_test: Vec<f32>,
}

impl<'a> TrainTestDataset<'a> {
    pub(crate) fn new(ds: &'a Dataset<f32, f32>) -> Self {
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

        Self {
            ds,
            x_train,
            y_train,
            x_test,
            y_test,
        }
    }
}

#[derive(Debug)]
pub(crate) struct DatasetParseError {
    pub(crate) msg: String
}

impl From<io::Error> for DatasetParseError {
    fn from(err: io::Error) -> Self {
        Self {
            msg: err.to_string()
        }
    }
}

impl From<ParseIntError> for DatasetParseError {
    fn from(err: ParseIntError) -> Self {
        Self {
            msg: err.to_string()
        }
    }
}

impl From<ParseFloatError> for DatasetParseError {
    fn from(err: ParseFloatError) -> Self {
        Self {
            msg: err.to_string()
        }
    }
}

impl From<Failed> for DatasetParseError {
    fn from(err: Failed) -> Self {
        Self {
            msg: err.to_string()
        }
    }
}

impl fmt::Display for DatasetParseError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let s = format!("error: {}", self.msg);
        write!(f, "{}", s)
    }
}