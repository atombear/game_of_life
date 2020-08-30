use ndarray_csv::{Array2Reader, Array2Writer};
use std::collections::HashMap;

/*
Any live cell with fewer than two live neighbours dies, as if by underpopulation.
Any live cell with two or three live neighbours lives on to the next generation.
Any live cell with more than three live neighbours dies, as if by overpopulation.
Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
*/

fn flip_board(brd: &mut ndarray::Array2<u8>, rows: &usize, cols: &usize) {
    for r in 0..*rows {
        for c in 0..*cols {
            let pos: [usize; 2] = [r, c];
            if brd[pos] == 1 {
                brd[pos] = 0;
            } else {
                brd[pos] = 1;
            }
        }
    }
}

fn gather_board_values(brd: &ndarray::Array2<u8>, pos_arr: &[(usize, usize)]) -> u8 {
    let mut ret: u8 = 0;
    for (r0, c0) in pos_arr.iter() {
        ret += brd[[*r0, *c0]];
    }
    return ret
}

fn count_neighbors(brd: &ndarray::Array2<u8>,
                   rows: &usize,
                   cols: &usize,
                   r: &usize,
                   c: &usize) -> u8 {
    // Upper left corner
    return if (*r == 0) & (*c == 0) { gather_board_values(brd, &[
        (0, 1),
        (1, 0),
        (1, 1)]) }
    // Upper right corner
    else if (*r == 0) & (*c == cols - 1) { gather_board_values(brd, &[
        (0, cols - 2),
        (1, cols - 2),
        (1, cols - 1)]) }
    // Bottom right corner
    else if (*r == rows - 1) & (*c == cols - 1) { gather_board_values(brd, &[
        (rows - 2, cols - 1),
        (rows - 2, cols - 2),
        (rows - 1, cols - 2)]) }
    // Bottom left corner
    else if (*r == rows - 1) & (*c == 0) { gather_board_values(brd, &[
        (rows - 2, 0),
        (rows - 2, 1),
        (rows - 1, 1)]) }
    // Top row
    else if *r == 0 { gather_board_values(brd, &[
        (0, c - 1),
        (1, c - 1),
        (1, *c),
        (1, c + 1),
        (0, c + 1)]) }
    // Right column
    else if *c == cols - 1 { gather_board_values(brd, &[
        (r - 1, *c),
        (r - 1, c - 1),
        (*r, c - 1),
        (r + 1, c - 1),
        (r + 1, *c)]) }
    // Bottom row
    else if *r == rows - 1 { gather_board_values(brd, &[
        (*r, c - 1),
        (r - 1, c - 1),
        (r - 1, *c),
        (r - 1, c + 1),
        (*r, c + 1)]) }
    //  Left column
    else if *c == 0 { gather_board_values(brd, &[
        (r - 1, 0),
        (r - 1, 1),
        (*r, 1),
        (r + 1, 1),
        (r + 1, 0)]) }
    // bulk
    else { gather_board_values(brd, &[
        (r - 1, c - 1),
        (r - 1, *c),
        (r - 1, c + 1),
        (*r, c + 1),
        (r + 1, c + 1),
        (r + 1, *c),
        (r + 1, c - 1),
        (*r, c - 1)]) }
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

    let rows: usize = data0.shape()[0];
    let cols: usize = data0.shape()[1];

    let mut boards: HashMap<u8, ndarray::Array2<u8>> = HashMap::new();
    boards.insert(0, data0);
    boards.insert(1, data1);

    flip_board(&mut boards.get_mut(&1).unwrap(), &rows, &cols);

    println!("{} {}", rows, cols);
    println!("{:?}", boards.get(&0).unwrap());
    println!("{:?}", boards.get(&1).unwrap());

    for i in 0..3 {
        for j in 0..3 {
            println!("{}", count_neighbors(&boards.get(&0).unwrap(), &rows, &cols, &i, &j));
        }
    }
}


