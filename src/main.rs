mod iris;
mod boston;
mod dataset;
mod wine;

fn main() {
    println!("Hello, world!");
    // iris::knn_classify();
    // iris::logistic_regression();
    // iris::guassian();
    // boston::linear_regression();
    // boston::logistic_regression();
    wine::linear_regression().expect("Error");
    wine::logistic_regression().expect("Error");
}
