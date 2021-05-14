mod iris;
mod boston;
mod dataset;
mod wine;

fn main() {
    // iris::knn_classify();
    // iris::logistic_regression();
    // iris::guassian();

    boston::knn_classify();
    boston::linear_regression();
    boston::logistic_regression();

    wine::knn_classify().expect("Error occurred in KNN");
    wine::linear_regression().expect("Error");
    wine::logistic_regression().expect("Error");
    wine::gaussian_regression().expect("Error");
}
