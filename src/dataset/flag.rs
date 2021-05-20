use std::{
    fs,
    io::{BufRead, BufReader},
};
use std::collections::HashMap;

use lazy_static;

use crate::dataset::DatasetParseError;
use smartcore::dataset::Dataset;

#[derive(Debug)]
pub(crate) struct Flag {
    name: String,
    landmass: u32,
    quadrant: u32,
    area: u32,
    population: u32,
    language: u32,
    religion: u32,
    bars: u32,
    stripes: u32,
    colours: u32,
    red: u32,
    green: u32,
    blue: u32,
    gold: u32,
    white: u32,
    black: u32,
    orange: u32,
    mainhue: String,
    circles: u32,
    crosses: u32,
    saltires: u32,
    quarters: u32,
    sunstars: u32,
    crescent: u32,
    triangle: u32,
    icon: u32,
    animate: u32,
    text: u32,
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
    fn features(&self) -> [u32; 11] {
        let mainhue = COLOURS.get(self.mainhue.as_str());
        let features = [
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
            *mainhue.unwrap()
        ];

        features
    }

    fn target(&self) -> u32 {
        self.religion
    }
}


fn parse_data_file(data_file: &str) -> Result<Vec<Flag>, DatasetParseError> {
    let dataf = fs::File::open(data_file)?;
    let data_lines = BufReader::new(dataf).lines();
    let mut flag_recs = Vec::<Flag>::new();

    for l in data_lines {
        if let Ok(s) = l {
            let attrs: Vec<&str> = s.split(",").collect();
            // let topleft = COLOURS.get(attrs[28]);
            // let botright = COLOURS.get(attrs[29]);

            let rec = Flag {
                name: attrs[0].to_string(),
                landmass: attrs[1].parse::<u32>()?,
                quadrant: attrs[2].parse::<u32>()?,
                area: attrs[3].parse::<u32>()?,
                population: attrs[4].parse::<u32>()?,
                language: attrs[5].parse::<u32>()?,
                religion: attrs[6].parse::<u32>()?,
                bars: attrs[7].parse::<u32>()?,
                stripes: attrs[8].parse::<u32>()?,
                colours: attrs[9].parse::<u32>()?,
                red: attrs[10].parse::<u32>()?,
                green: attrs[11].parse::<u32>()?,
                blue: attrs[12].parse::<u32>()?,
                gold: attrs[13].parse::<u32>()?,
                white: attrs[14].parse::<u32>()?,
                black: attrs[15].parse::<u32>()?,
                orange: attrs[16].parse::<u32>()?,
                mainhue: attrs[17].to_string(),
                circles: attrs[18].parse::<u32>()?,
                crosses: attrs[19].parse::<u32>()?,
                saltires: attrs[20].parse::<u32>()?,
                quarters: attrs[21].parse::<u32>()?,
                sunstars: attrs[22].parse::<u32>()?,
                crescent: attrs[23].parse::<u32>()?,
                triangle: attrs[24].parse::<u32>()?,
                icon: attrs[25].parse::<u32>()?,
                animate: attrs[26].parse::<u32>()?,
                text: attrs[27].parse::<u32>()?,
                topleft: attrs[28].to_string(),
                botright: attrs[29].to_string()
            };

            flag_recs.push(rec);
        }
    }

    Ok(flag_recs)
}

pub(crate) fn load_dataset(data_file: &str) -> Result<Dataset<u32, u32>, DatasetParseError> {
    let flag_recs = parse_data_file(data_file)?;
    let mut x: Vec<u32> = Vec::new();
    let mut y: Vec<u32> = Vec::new();
    let num_samples = flag_recs.len();
    let feature_names = vec![
        "bars", "stripes", "colours", "red",
        "green", "blue", "gold", "white",
        "black", "orange", "mainhue"
    ];

    for rec in flag_recs {
        let mut rec_features = rec.features().to_vec();
        x.append(&mut rec_features);
        y.push(rec.target());
    }

    let ds = Dataset {
        data: x,
        target: y,
        num_samples,
        num_features: feature_names.len(),
        feature_names: feature_names
            .iter()
            .map(|s| s.to_string())
            .collect(),
        target_names: vec!["religion".to_string()],
        description: "The flags data: https://archive.ics.uci.edu/ml/datasets/Flags"
            .to_string(),
    };
    println!("ds: samples={}, features={}, target={}",
             ds.num_samples, ds.num_features, ds.target_names.join(", "));

    Ok(ds)
}

#[cfg(test)]
mod tests {

    use super::load_dataset;
    use crate::dataset::FLAG_DATASET;

    const FLAG_DATASET_SAMPLES_SIZE: usize = 194;
    const FLAG_DATASET_NO_FEATURES: usize = 11;

    #[test]
    fn test_flag_dataset() {
        let ds = load_dataset(FLAG_DATASET)
            .expect("Failed to load flag dataset!");
        assert_eq!(ds.data.len(), FLAG_DATASET_SAMPLES_SIZE * FLAG_DATASET_NO_FEATURES);
        assert_eq!(ds.target.len(), FLAG_DATASET_SAMPLES_SIZE);

        // check first column of first sample
        assert_eq!(ds.data[0], 0);
        // check first column of last sample
        assert_eq!(ds.data[(FLAG_DATASET_SAMPLES_SIZE-1)*FLAG_DATASET_NO_FEATURES], 0);
    }
}