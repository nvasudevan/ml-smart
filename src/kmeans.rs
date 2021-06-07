use smartcore::metrics::{completeness_score, homogeneity_score};
use crate::results::KMeansResult;
use smartcore::cluster::kmeans::{KMeansParameters, KMeans};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;

pub(crate) fn run_n_k(dm: &DenseMatrix<f32>, n: usize, k: usize, labels_true: &Vec<f32>) -> KMeansResult {
    let model = KMeans::fit(
        dm, KMeansParameters::default()
            .with_k(k).with_max_iter(n),
    ).expect("Failed to create a model");
    let labels_pred = model.predict(dm)
        .expect("Unable to predict");

    KMeansResult::new(n, k,
                      homogeneity_score(labels_true, &labels_pred),
                      completeness_score(labels_true, &labels_pred),
    )
}

