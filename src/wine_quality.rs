use crate::dataset;

pub(crate) fn test() {
    let red_ds = dataset::wine_quality::load_red_dataset();
    let white_ds = dataset::wine_quality::load_white_dataset();
}