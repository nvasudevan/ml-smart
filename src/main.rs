mod iris;
mod boston;
mod dataset;
mod wine;
mod wine_quality;

// applies to all datasets
pub(crate) const TRAINING_TEST_SIZE_RATIO: f32 = 0.7; // train=20%; test=80%

fn main() {
    // iris::knn_classify();
    // iris::logistic_regression();
    // iris::guassian();

    // boston::knn_classify();
    // boston::linear_regression();
    // boston::logistic_regression();
    //
    // wine::knn_classify().expect("Error occurred in KNN");
    // wine::linear_regression().expect("Error");
    // wine::logistic_regression().expect("Error");
    // wine::gaussian_regression().expect("Error");

    wine_quality::knn_classify()
        .expect("Error whilst running kNN classifier for wine quality");
    wine_quality::logistic_regression()
        .expect("Error whilst running logistic for wine quality");
    // wine_quality::gaussian_regression()
    //     .expect("Error whilst running Gaussian NB classifier for wine quality");
    wine_quality::linear_regression()
        .expect("Error whilst running linear for wine quality");
}
