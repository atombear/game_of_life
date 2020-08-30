use ndarray_csv::{Array2Reader, Array2Writer};
use std::collections::HashMap;

fn flip_board(brd: &mut ndarray::Array2<u8>, row: &usize, col: &usize) {
    for r in 0..*row {
        for c in 0..*col {
            if brd[[r, c]] == 1 {
                brd[[r, c]] = 0;
            } else {
                brd[[r, c]] = 1;
            }
        }
    }
}

fn main() {

    let mut csv_path: std::path::PathBuf = std::env::current_exe().unwrap();
    for _ in 0..3 { csv_path.pop(); }
    for val in vec!["src", "board.csv"] { csv_path.push(val); }

    let mut reader = csv::ReaderBuilder::new().has_headers(false)
                                              .from_path(csv_path)
                                              .expect("Cannot read file");

    let mut data0: ndarray::Array2<u8> = reader.deserialize_array2_dynamic().unwrap();
    let mut data1: ndarray::Array2<u8> = data0.to_owned();

    let row: usize = data0.shape()[0];
    let col: usize = data0.shape()[1];

    let mut boards: HashMap<u8, ndarray::Array2<u8>> = HashMap::new();
    boards.insert(0, data0);
    boards.insert(1, data1);

    flip_board(&mut boards.get_mut(&1).unwrap(), &row, &col);

    println!("{} {}", row, col);
    println!("{:?}", boards.get(&0).unwrap());
    println!("{:?}", boards.get(&1).unwrap());
}


