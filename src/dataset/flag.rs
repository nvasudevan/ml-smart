use std::{
    fs,
    io::{BufRead, BufReader},
};
use std::collections::HashMap;

use lazy_static;

use crate::dataset::DatasetParseError;
use smartcore::dataset::Dataset;

pub(crate) const FEATURE_TGT_RELIGION: [&str; 22] = [
    "landmass", "language", "bars", "stripes", "colours", "red",
    "green", "blue", "gold", "white",
    "black", "orange", "mainhue", "circles",
    "crosses", "saltires", "sunstars", "crescent",
    "triangle", "icon", "topleft", "botright"
];

pub(crate) const TGT_RELIGION: [&str; 1] = ["religion"];

pub(crate) const FEATURE_TGT_LANGUAGE: [&str; 22] = [
    "landmass", "religion", "bars", "stripes", "colours", "red",
    "green", "blue", "gold", "white",
    "black", "orange", "mainhue", "circles",
    "crosses", "saltires", "sunstars", "crescent",
    "triangle", "icon", "topleft", "botright"
];

pub(crate) const TGT_LANGUAGE: [&str; 1] = ["language"];

#[derive(Debug, Clone)]
pub(crate) struct Flag {
    name: String,
    landmass: f32,
    quadrant: f32,
    area: f32,
    population: f32,
    language: f32,
    religion: f32,
    bars: f32,
    stripes: f32,
    colours: f32,
    red: f32,
    green: f32,
    blue: f32,
    gold: f32,
    white: f32,
    black: f32,
    orange: f32,
    mainhue: String,
    circles: f32,
    crosses: f32,
    saltires: f32,
    quarters: f32,
    sunstars: f32,
    crescent: f32,
    triangle: f32,
    icon: f32,
    animate: f32,
    text: f32,
    topleft: String,
    botright: String,
}

lazy_static! {
    static ref LANDMASS: HashMap<u32, &'static str> = {
        let mut landmass_map = HashMap::new();
        landmass_map.insert(1, "N. America");
        landmass_map.insert(2, "S. America");
        landmass_map.insert(3, "Europe");
        landmass_map.insert(4, "Africa");
        landmass_map.insert(5, "Asia");
        landmass_map.insert(6, "Oceania");

        landmass_map
    };

    static ref QUADRANT: HashMap<u32, &'static str> = {
        let mut quadrant_map = HashMap::new();
        quadrant_map.insert(1, "NE");
        quadrant_map.insert(2, "SE");
        quadrant_map.insert(3, "SW");
        quadrant_map.insert(4, "NW");

        quadrant_map
    };

    static ref LINGUA: HashMap<u32, &'static str> = {
        let mut lingua_map = HashMap::new();
        lingua_map.insert(1, "English");
        lingua_map.insert(2, "Spanish");
        lingua_map.insert(3, "French");
        lingua_map.insert(4, "German");
        lingua_map.insert(5, "Slavic");
        lingua_map.insert(6, "Other Indo-European");
        lingua_map.insert(7, "Chinese");
        lingua_map.insert(8, "Arabic");
        lingua_map.insert(9, "Japanese/Turkish/Finnish/Magyar");
        lingua_map.insert(10, "Others");

        lingua_map
    };

    static ref RELIGION: HashMap<u32, &'static str> = {
        let mut religion_map = HashMap::new();
        religion_map.insert(0, "Catholic");
        religion_map.insert(1, "Other Christian");
        religion_map.insert(2, "Muslim");
        religion_map.insert(3, "Buddhist");
        religion_map.insert(4, "Hindu");
        religion_map.insert(5, "Ethnic");
        religion_map.insert(6, "Marxist");
        religion_map.insert(7, "Others");

        religion_map
    };

    static ref COLOURS: HashMap<&'static str, u32> = {
        let mut colours_map = HashMap::new();
        colours_map.insert("black", 1);
        colours_map.insert("blue", 2);
        colours_map.insert("brown", 3);
        colours_map.insert("gold", 4);
        colours_map.insert("green", 5);
        colours_map.insert("orange", 6);
        colours_map.insert("red", 7);
        colours_map.insert("white", 8);

        colours_map
    };
}

impl Flag {
    // features and target for prediction religion
    fn features_tgt_religion(&self) -> [f32; 22] {
        let mainhue = COLOURS.get(self.mainhue.as_str())
            .expect("Unable to parse mainhue as integer");
        let topleft = COLOURS.get(self.topleft.as_str())
            .expect("Unable to parse topleft as integer");
        let botright = COLOURS.get(self.botright.as_str())
            .expect("Unable to parse botright as integer");
        let features = [
            self.landmass,
            self.language,
            self.bars,
            self.stripes,
            self.colours,
            self.red,
            self.green,
            self.blue,
            self.gold,
            self.white,
            self.black,
            self.orange,
            *mainhue as f32,
            self.circles,
            self.crosses,
            self.saltires,
            self.sunstars,
            self.crescent,
            self.triangle,
            self.icon,
            *topleft as f32,
            *botright as f32
        ];

        features
    }

    fn tgt_religion(&self) -> f32 {
        self.religion
    }

    // features and target for prediction language
    fn features_tgt_language(&self) -> [f32; 22] {
        let mainhue = COLOURS.get(self.mainhue.as_str())
            .expect("Unable to parse mainhue as integer");
        let topleft = COLOURS.get(self.topleft.as_str())
            .expect("Unable to parse topleft as integer");
        let botright = COLOURS.get(self.botright.as_str())
            .expect("Unable to parse botright as integer");
        let features = [
            self.landmass,
            self.religion,
            self.bars,
            self.stripes,
            self.colours,
            self.red,
            self.green,
            self.blue,
            self.gold,
            self.white,
            self.black,
            self.orange,
            *mainhue as f32,
            self.circles,
            self.crosses,
            self.saltires,
            self.sunstars,
            self.crescent,
            self.triangle,
            self.icon,
            *topleft as f32,
            *botright as f32
        ];

        features
    }

    fn tgt_language(&self) -> f32 {
        self.language
    }

    pub fn country(&self) -> String {
        self.name.to_string()
    }
}

fn parse_data_file(data_file: &str) -> Result<Vec<Flag>, DatasetParseError> {
    let dataf = fs::File::open(data_file)?;
    let data_lines = BufReader::new(dataf).lines();
    let mut flag_recs = Vec::<Flag>::new();

    for l in data_lines {
        if let Ok(s) = l {
            let attrs: Vec<&str> = s.split(",").collect();

            let rec = Flag {
                name: attrs[0].to_string(),
                landmass: attrs[1].parse::<f32>()?,
                quadrant: attrs[2].parse::<f32>()?,
                area: attrs[3].parse::<f32>()?,
                population: attrs[4].parse::<f32>()?,
                language: attrs[5].parse::<f32>()?,
                religion: attrs[6].parse::<f32>()?,
                bars: attrs[7].parse::<f32>()?,
                stripes: attrs[8].parse::<f32>()?,
                colours: attrs[9].parse::<f32>()?,
                red: attrs[10].parse::<f32>()?,
                green: attrs[11].parse::<f32>()?,
                blue: attrs[12].parse::<f32>()?,
                gold: attrs[13].parse::<f32>()?,
                white: attrs[14].parse::<f32>()?,
                black: attrs[15].parse::<f32>()?,
                orange: attrs[16].parse::<f32>()?,
                mainhue: attrs[17].to_string(),
                circles: attrs[18].parse::<f32>()?,
                crosses: attrs[19].parse::<f32>()?,
                saltires: attrs[20].parse::<f32>()?,
                quarters: attrs[21].parse::<f32>()?,
                sunstars: attrs[22].parse::<f32>()?,
                crescent: attrs[23].parse::<f32>()?,
                triangle: attrs[24].parse::<f32>()?,
                icon: attrs[25].parse::<f32>()?,
                animate: attrs[26].parse::<f32>()?,
                text: attrs[27].parse::<f32>()?,
                topleft: attrs[28].to_string(),
                botright: attrs[29].to_string(),
            };

            flag_recs.push(rec);
        }
    }

    Ok(flag_recs)
}

pub(crate) fn load_dataset_tgt_religion(data_file: &str) -> Result<(Vec<Flag>, Dataset<f32, f32>), DatasetParseError> {
    let flag_recs = parse_data_file(data_file)?;
    let mut x: Vec<f32> = Vec::new();
    let mut y: Vec<f32> = Vec::new();
    let num_samples = flag_recs.len();
    for rec in &flag_recs {
        let mut rec_features = rec.features_tgt_religion().to_vec();
        x.append(&mut rec_features);
        y.push(rec.tgt_religion());
    }

    let ds = Dataset {
        data: x,
        target: y,
        num_samples,
        num_features: FEATURE_TGT_RELIGION.len(),
        feature_names: FEATURE_TGT_RELIGION
            .iter()
            .map(|s| s.to_string())
            .collect(),
        target_names: TGT_RELIGION
            .iter()
            .map(|s| s.to_string())
            .collect(),
        description: "The flags data: https://archive.ics.uci.edu/ml/datasets/Flags"
            .to_string(),
    };
    println!("ds: samples={}, features={}, target={}",
             ds.num_samples, ds.num_features, ds.target_names.join(", "));

    Ok((flag_recs, ds))
}

pub(crate) fn load_dataset_tgt_language(data_file: &str) -> Result<(Vec<Flag>, Dataset<f32, f32>), DatasetParseError> {
    let flag_recs = parse_data_file(data_file)?;
    let mut x: Vec<f32> = Vec::new();
    let mut y: Vec<f32> = Vec::new();
    let num_samples = flag_recs.len();
    for rec in &flag_recs {
        let mut rec_features = rec.features_tgt_language().to_vec();
        x.append(&mut rec_features);
        y.push(rec.tgt_language());
    }

    let ds = Dataset {
        data: x,
        target: y,
        num_samples,
        num_features: FEATURE_TGT_LANGUAGE.len(),
        feature_names: FEATURE_TGT_LANGUAGE
            .iter()
            .map(|s| s.to_string())
            .collect(),
        target_names: TGT_LANGUAGE
            .iter()
            .map(|s| s.to_string())
            .collect(),
        description: "The flags data: https://archive.ics.uci.edu/ml/datasets/Flags"
            .to_string(),
    };
    println!("ds: samples={}, features={}, target={}",
             ds.num_samples, ds.num_features, ds.target_names.join(", "));

    Ok((flag_recs, ds))
}

#[cfg(test)]
mod tests {
    use super::load_dataset_tgt_religion;
    use crate::dataset::FLAG_DATASET;

    const FLAG_DATASET_SAMPLES_SIZE: usize = 194;
    const FLAG_DATASET_NO_FEATURES: usize = 11;

    #[test]
    fn test_flag_dataset() {
        let ds = load_dataset_tgt_religion(FLAG_DATASET)
            .expect("Failed to load flag dataset!");
        assert_eq!(ds.data.len(), FLAG_DATASET_SAMPLES_SIZE * FLAG_DATASET_NO_FEATURES);
        assert_eq!(ds.target.len(), FLAG_DATASET_SAMPLES_SIZE);

        // check first column of first sample
        assert_eq!(ds.data[0], 0.0);
        // check first column of last sample
        assert_eq!(ds.data[(FLAG_DATASET_SAMPLES_SIZE - 1) * FLAG_DATASET_NO_FEATURES], 0.0);
    }
}