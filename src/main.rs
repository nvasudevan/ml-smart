use crate::results::MLResult;
use prettytable::Table;

#[macro_use]
extern crate prettytable;

#[macro_use]
extern crate lazy_static;

mod iris;
mod boston;
mod dataset;
mod wine;
mod wine_quality;
mod results;
mod flags;

// applies to all datasets
pub(crate) const TRAINING_TEST_SIZE_RATIO: f32 = 0.7; // train=20%; test=80%
pub(crate) const MAX_NO_CHANGES: usize = 20;

pub(crate) fn show(results: Vec<MLResult>) {
    let mut table = Table::new();
    table.add_row(row!["ML algo", "accuracy", "MAE", "MSE*"]);
    for ml in results {
        table.add_row(row![ml.name(), ml.acc(), ml.mae(), ml.mse()]);
    }

    table.printstd();
    println!();
}

fn main() {
    // let iris_results = iris::run()
    //     .expect("ML run failed for Iris dataset");
    // println!("\n=>[IRIS] ML result: \n");
    // show(iris_results);
    //
    // let boston_results = boston::run()
    //     .expect("ML run failed for Boston dataset");
    // println!("\n=>[Boston] ML result: \n");
    // show(boston_results);
    //
    // let wine_results = wine::run()
    //     .expect("Failed to run wine dataset");
    // println!("\n=>[Wine class] ML result: \n");
    // show(wine_results);

    // let red_wine_quality_results = wine_quality::run_red()
    //     .expect("Failed to run red wine quality dataset");
    // println!("\n=>[Red Wine quality] ML result: \n");
    // show(red_wine_quality_results);
    //
    // let white_wine_quality_results = wine_quality::run_white()
    //     .expect("Failed to run white wine quality dataset");
    // println!("\n=>[White Wine quality] ML result: \n");
    // show(white_wine_quality_results);

    let flag_re_results = flags::run_predict_religion()
        .expect("Failed to run flag dataset");
    println!("\n=>[Flag][predict: religion] ML result\n({}) => {}: \n",
             dataset::flag::FEATURE_TGT_RELIGION.join(","),
             dataset::flag::TGT_RELIGION.join(","));
    show(flag_re_results);

    // let flag_lang_results = flags::run_predict_language()
    //     .expect("Failed to run flag dataset");
    // println!("\n=>[Flag][predict: language] ML result\n({}) => {}: \n",
    //          dataset::flag::FEATURE_TGT_LANGUAGE.join(","),
    //          dataset::flag::TGT_LANGUAGE.join(","));
    // show(flag_lang_results);
}
