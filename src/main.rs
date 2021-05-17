use crate::results::MLResult;
use prettytable::Table;

#[macro_use]
extern crate prettytable;

mod iris;
mod boston;
mod dataset;
mod wine;
mod wine_quality;
mod results;

// applies to all datasets
pub(crate) const TRAINING_TEST_SIZE_RATIO: f32 = 0.7; // train=20%; test=80%

pub(crate) fn show(results: Vec<MLResult>) {
    let mut table = Table::new();
    table.add_row(row!["ML algo", "accuracy", "MAE"]);
    for ml in results {
        table.add_row(row![ml.name(), ml.acc(), ml.mae()]);
    }

    table.printstd();
}

fn main() {
    let iris_results = iris::run();
    println!("\n=>[IRIS] ML result: \n");
    show(iris_results);

    let boston_results = boston::run();
    println!("\n=>[Boston] ML result: \n");
    show(boston_results);

    // boston::knn_classify();
    // boston::knn_regression();
    // boston::linear_regression();
    // boston::logistic_regression();
    // // boston::gaussianNB();
    // boston::categoricalNB();

    // wine::knn_classify().expect("Error occurred in KNN");
    // wine::knn_regression().expect("Error occurred in kNN");
    // wine::linear_regression().expect("Error");
    // wine::logistic_regression().expect("Error");
    // wine::gaussian_regression().expect("Error");
    // wine::categoricalNB();
    // wine::multinomialNB();

    // wine_quality::knn_classify()
    //     .expect("Error whilst running kNN classifier for wine quality");
    // wine_quality::knn_regression()
    //     .expect("Error whilst running kNN regression for wine quality");
    // wine_quality::logistic_regression()
    //     .expect("Error whilst running logistic for wine quality");
    // // wine_quality::gaussian_regression()
    // //     .expect("Error whilst running Gaussian NB classifier for wine quality");
    // wine_quality::linear_regression()
    //     .expect("Error whilst running linear for wine quality");
    // wine_quality::categoricalNB()
    //     .expect("Error whilst running categorical NB for wine quality");
    // wine_quality::multinomialNB()
    //     .expect("Error whilst running multinomial NB for wine quality");
}
