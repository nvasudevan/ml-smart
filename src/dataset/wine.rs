use std::{fs, io};
use std::io::{Read, BufReader, BufRead};
use crate::dataset::DatasetParseError;
use smartcore::dataset::Dataset;

const WINE_DATASET_SAMPLES_SIZE: usize = 178;
const WINE_DATASET_NO_FEATURES: usize = 13;

#[derive(Debug)]
pub(crate) struct WineRecord {
    pub wine_class: f32,
    pub alcohol: f32,
    pub malic_acid: f32,
    pub ash: f32,
    pub ash_alkalinity: f32,
    pub magnesium: f32,
    pub total_phenols: f32,
    pub flavanoids: f32,
    pub non_flavonoid_phenols: f32,
    pub proanthocynanins: f32,
    pub colour_intensity: f32,
    pub hue: f32,
    pub od: f32,
    pub proline: f32,
}

impl WineRecord {
    fn features(&self) -> [f32;13] {
        let features: [f32; 13] = [
            self.alcohol,
            self.malic_acid,
            self.ash,
            self.ash_alkalinity,
            self.magnesium,
            self.total_phenols,
            self.flavanoids,
            self.non_flavonoid_phenols,
            self.proanthocynanins,
            self.colour_intensity,
            self.hue,
            self.od,
            self.proline
        ];

        features
    }

    fn target(&self) -> f32 {
        self.wine_class
    }
}

fn parse_data_file(data_file: &str) -> Result<Vec<WineRecord>, DatasetParseError> {
    let mut dataf = fs::File::open(data_file)?;
    let data_lines = BufReader::new(dataf).lines();
    let mut wine_recs = Vec::<WineRecord>::new();

    for l in data_lines {
        if let Ok(s) = l {
            let attrs: Vec<&str> = s.split(",").collect();
            let wine_class = attrs[0].parse::<f32>()?;
            let alcohol = attrs[1].parse::<f32>()?;
            let malic_acid = attrs[2].parse::<f32>()?;
            let ash = attrs[3].parse::<f32>()?;
            let ash_alkalinity = attrs[4].parse::<f32>()?;
            let magnesium = attrs[5].parse::<f32>()?;
            let total_phenols = attrs[6].parse::<f32>()?;
            let flavanoids = attrs[7].parse::<f32>()?;
            let non_flavonoid_phenols = attrs[8].parse::<f32>()?;
            let proanthocynanins = attrs[9].parse::<f32>()?;
            let colour_intensity = attrs[10].parse::<f32>()?;
            let hue = attrs[11].parse::<f32>()?;
            let od = attrs[12].parse::<f32>()?;
            let proline = attrs[13].parse::<f32>()?;

            let rec = WineRecord {
                wine_class,
                alcohol,
                malic_acid,
                ash,
                ash_alkalinity,
                magnesium,
                total_phenols,
                flavanoids,
                non_flavonoid_phenols,
                proanthocynanins,
                colour_intensity,
                hue,
                od,
                proline,
            };

            wine_recs.push(rec);
        }
    }

    Ok(wine_recs)
}

pub(crate) fn load_dataset(data_file: &str) -> Result<Dataset<f32, f32>, DatasetParseError> {
    let wine_records = parse_data_file(data_file)?;
    let mut x: Vec<f32> = Vec::new();
    let mut y: Vec<f32> = Vec::new();

    for rec in wine_records {
        let mut rec_features = rec.features().to_vec();
        x.append(&mut rec_features);
        y.push(rec.target());
    }

    let ds = Dataset {
        data: x,
        target: y,
        num_samples: WINE_DATASET_SAMPLES_SIZE,
        num_features: WINE_DATASET_NO_FEATURES,
        feature_names: vec![
            "Alcohol", "Malic Acid", "Ash", "Ash Alkalinity", "Magnesium",
            "Total Phenols", "Flavonoids", "Non-Flavonoid Phenols", "Proanthocynanins",
            "Colour Intensity", "Hue", "OD", "Proline",
        ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        target_names: vec!["wine-class".to_string()],
        description: "The Boston house-price data: https://archive.ics.uci.edu/ml/datasets/Wine"
            .to_string(),
    };
    println!("ds: samples={}, features={}, target={}",
             ds.num_samples, ds.num_features, ds.target_names.join(", "));

    Ok(ds)
}

#[cfg(test)]
mod tests {

    use super::{Dataset, load_dataset};
    use crate::dataset::wine::{WINE_DATASET_SAMPLES_SIZE, WINE_DATASET_NO_FEATURES};

    #[test]
    fn test_dataset() {
        let ds = load_dataset(crate::dataset::WINE_DATASET)
            .expect("Unable to load the wine dataset");
        assert_eq!(ds.data.len(), WINE_DATASET_SAMPLES_SIZE * WINE_DATASET_NO_FEATURES);
        assert_eq!(ds.target.len(), WINE_DATASET_SAMPLES_SIZE);

        // check first column of first sample
        assert_eq!(ds.data[0], 14.23);
        // check first column of last sample
        // so look at 177th record * 13 features => first column of 178 record
        assert_eq!(ds.data[(WINE_DATASET_SAMPLES_SIZE-1)*WINE_DATASET_NO_FEATURES], 14.13);
    }
}