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
    println!();
}

fn main() {
    let iris_results = iris::run()
        .expect("ML run failed for Iris dataset");
    println!("\n=>[IRIS] ML result: \n");
    show(iris_results);

    let boston_results = boston::run()
        .expect("ML run failed for Boston dataset");
    println!("\n=>[Boston] ML result: \n");
    show(boston_results);

    let wine_results = wine::run()
        .expect("Failed to run wine dataset");
    println!("\n=>[Wine class] ML result: \n");
    show(wine_results);

    let red_wine_quality_results = wine_quality::run_red()
        .expect("Failed to run red wine quality dataset");
    println!("\n=>[Red Wine quality] ML result: \n");
    show(red_wine_quality_results);

    let white_wine_quality_results = wine_quality::run_white()
        .expect("Failed to run white wine quality dataset");
    println!("\n=>[White Wine quality] ML result: \n");
    show(white_wine_quality_results);
}
