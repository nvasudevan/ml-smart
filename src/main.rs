mod iris;
mod boston;
mod dataset;
mod wine;

fn main() {
    println!("Hello, world!");
    // iris::knn_classify();
    // iris::regression();
    // iris::guassian();
    // boston::logistic_regression();
    // boston::linear_regression();
    wine::linear_regression()
        .expect("Error");
}
