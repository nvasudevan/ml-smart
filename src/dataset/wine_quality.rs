use crate::dataset::{
    DatasetParseError,
    WINE_RED_QUALITY_DATASET,
    WINE_WHITE_QUALITY_DATASET,
};

use std::io::{BufReader, BufRead};
use std::fs;
use smartcore::dataset::Dataset;

#[derive(Debug)]
struct WineQuality {
    fixed_acidity: f32,
    volatile_acidity: f32,
    citric_acid: f32,
    residual_sugar: f32,
    chlorides: f32,
    free_sulfur_dioxide: f32,
    total_sulfur_dioxide: f32,
    density: f32,
    pH: f32,
    sulphates: f32,
    alcohol: f32,
    quality: f32,
}

impl WineQuality {
    fn features(&self) -> [f32; 11] {
        let features: [f32; 11] = [
            self.fixed_acidity,
            self.volatile_acidity,
            self.citric_acid,
            self.residual_sugar,
            self.chlorides,
            self.free_sulfur_dioxide,
            self.total_sulfur_dioxide,
            self.density,
            self.pH,
            self.sulphates,
            self.alcohol,
        ];

        features
    }

    fn target(&self) -> f32 {
        self.quality
    }
}

fn parse_data_file(data_file: &str) -> Result<(Vec<String>, Vec<WineQuality>), DatasetParseError> {
    let dataf = fs::File::open(data_file)?;
    let data_lines = BufReader::new(dataf).lines();
    let mut wine_recs = Vec::<WineQuality>::new();

    let mut headers = Vec::<String>::new();
    for (n, l) in data_lines.enumerate() {
        if let Ok(s) = l {
            if n == 0 {
                headers = s.split(";").map(|v| v.to_string()).collect();
            } else {
                let attrs: Vec<&str> = s.split(";").collect();
                let fixed_acidity = attrs[0].parse::<f32>()?;
                let volatile_acidity = attrs[1].parse::<f32>()?;
                let citric_acid = attrs[2].parse::<f32>()?;
                let residual_sugar = attrs[3].parse::<f32>()?;
                let chlorides = attrs[4].parse::<f32>()?;
                let free_sulfur_dioxide = attrs[5].parse::<f32>()?;
                let total_sulfur_dioxide = attrs[6].parse::<f32>()?;
                let density = attrs[7].parse::<f32>()?;
                let pH = attrs[8].parse::<f32>()?;
                let sulphates = attrs[9].parse::<f32>()?;
                let alcohol = attrs[10].parse::<f32>()?;
                let quality = attrs[11].parse::<f32>()?;

                let rec = WineQuality {
                    fixed_acidity,
                    volatile_acidity,
                    citric_acid,
                    residual_sugar,
                    chlorides,
                    free_sulfur_dioxide,
                    total_sulfur_dioxide,
                    density,
                    pH,
                    alcohol,
                    sulphates,
                    quality,
                };

                wine_recs.push(rec);
            }
        }
    }

    Ok((headers, wine_recs))
}

fn load_dataset(data_file: &str) -> Result<Dataset<f32, f32>, DatasetParseError> {
    let (mut headers, wine_records) = parse_data_file(data_file)?;
    let mut x: Vec<f32> = Vec::new();
    let mut y: Vec<f32> = Vec::new();
    let num_samples = wine_records.len();

    for rec in wine_records {
        let mut rec_features = rec.features().to_vec();
        x.append(&mut rec_features);
        y.push(rec.target());
    }

    // the last column is the target
    let target_name = headers.pop().unwrap();
    let num_features = headers.len();
    let ds = Dataset {
        data: x,
        target: y,
        num_samples,
        num_features,
        feature_names: headers,
        target_names: vec![target_name],
        description: "The Wine quality data: https://archive.ics.uci.edu/ml/datasets/Wine+Quality"
            .to_string(),
    };
    println!("ds: samples={}, features={}, target={}",
             ds.num_samples, ds.num_features, ds.target_names.join(", "));

    Ok(ds)
}

pub(crate) fn load_red_dataset() -> Result<Dataset<f32, f32>, DatasetParseError> {
    load_dataset(WINE_RED_QUALITY_DATASET)
}

pub(crate) fn load_white_dataset() -> Result<Dataset<f32, f32>, DatasetParseError> {
    load_dataset(WINE_WHITE_QUALITY_DATASET)
}

#[cfg(test)]
mod tests {

    use super::load_red_dataset;
    use super::load_white_dataset;

    const WINE_RED_DATASET_SAMPLES_SIZE: usize = 1599;
    const WINE_RED_DATASET_NO_FEATURES: usize = 11;
    const WINE_WHITE_DATASET_SAMPLES_SIZE: usize = 4898;
    const WINE_WHITE_DATASET_NO_FEATURES: usize = 11;

    #[test]
    fn test_red_dataset() {
        let ds = load_red_dataset()
            .unwrap();
        assert_eq!(ds.data.len(), WINE_RED_DATASET_SAMPLES_SIZE * WINE_RED_DATASET_NO_FEATURES);
        assert_eq!(ds.target.len(), WINE_RED_DATASET_SAMPLES_SIZE);

        // check first column of first sample
        assert_eq!(ds.data[0], 7.4);
        // check first column of last sample
        assert_eq!(ds.data[(WINE_RED_DATASET_SAMPLES_SIZE-1)*WINE_RED_DATASET_NO_FEATURES], 6.0);
    }

    #[test]
    fn test_white_dataset() {
        let ds = load_white_dataset()
            .unwrap();
        assert_eq!(ds.data.len(), WINE_WHITE_DATASET_SAMPLES_SIZE * WINE_WHITE_DATASET_NO_FEATURES);
        assert_eq!(ds.target.len(), WINE_WHITE_DATASET_SAMPLES_SIZE);

        // check first column of first sample
        assert_eq!(ds.data[0], 7.0);
        // check first column of last sample
        assert_eq!(ds.data[(WINE_WHITE_DATASET_SAMPLES_SIZE-1)*WINE_WHITE_DATASET_NO_FEATURES], 6.0);
    }
}
