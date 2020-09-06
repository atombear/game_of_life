use ndarray_csv::{Array2Reader, Array2Writer};
use std::{cell::RefCell, collections::HashMap, thread::sleep, time::Duration, thread};

static NUM_ROW_GROUPS: u64 = 3;
static NUM_COL_GROUPS: u64 = 3;

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
                brd[pos] = 0; }
            else {
                brd[pos] = 1; }
        }
    }
}

fn gather_board_values(brd: &ndarray::Array2<u8>, pos_arr: &[(usize, usize)]) -> u8 {
    let mut ret: u8 = 0;
    for (r0, c0) in pos_arr.iter() { ret += brd[[*r0, *c0]]; }
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

fn print_board(brd: &ndarray::Array2<u8>,
               rows: &usize,
               cols: &usize) {
    for r in 0..*rows {
        for c in 0..*cols {
            print!("{} ", brd[[r, c]]);
        }
        print!("{}", "\n");
    }
}

fn capture_moves(brd: &ndarray::Array2<u8>,
                 rows: &usize,
                 cols: &usize,
                 start_row: &usize,
                 stop_row: &usize,
                 start_col: &usize,
                 stop_col: &usize,
                 moves: &mut Vec<(usize, usize, u8)>) {
    for r in *start_row..*stop_row {
        for c in *start_col..*stop_col {
            let count = count_neighbors(brd, rows, cols, &r, &c);
            if (brd[[r, c]] == 1) & ((count < 2) | (count > 3)) {
                moves.push((r, c, 0)); }
            else if count == 3 {
                moves.push((r, c, 1)); }
        }
    }
}

fn get_groups(num: u64, num_groups: u64) -> Vec<usize> {
    let mut ret: Vec<usize> = vec![];
    let mut rem = num - num_groups * (num / num_groups);

    for _ in 0..num_groups {
        ret.push((num / num_groups) as usize);
        if rem > 0 {
            let len = ret.len();
            ret[len - 1] += 1;
            rem -= 1;
        }
    }
    return ret
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

    let mut boards: HashMap<u8, RefCell<ndarray::Array2<u8>>> = HashMap::new();
    boards.insert(0, RefCell::new(data0));
    boards.insert(1, RefCell::new(data1));

    // Track the changes that must be made to the next iteration.
    let mut moves: Vec<(usize, usize, u8)> = vec![];
    // A board for visualization and a board for modification.
    let viz_board = boards.get(&0).unwrap().borrow();
    let mut life_board = boards.get(&1).unwrap().borrow_mut();

    let row_groups: Vec<usize> = get_groups(rows as u64, NUM_ROW_GROUPS);
    let col_groups: Vec<usize> = get_groups(cols as u64, NUM_COL_GROUPS);

    let mut extents: Vec<(usize, usize, usize, usize)> = vec![];

    let mut row_first_idx: usize = 0;
    for r_len in &row_groups {
        let mut col_first_idx: usize = 0;
        for c_len in &col_groups {
            extents.push((row_first_idx,
                          row_first_idx + r_len,
                          col_first_idx,
                          col_first_idx + c_len));
            col_first_idx += c_len;
        }
        row_first_idx += r_len;
    }

    // let handles: Vec<thread::JoinHandle<_>> = vec![];
    for iter in 0..50 {

        print!("{}[2J", 27 as char);
        sleep(Duration::from_millis(500 as u64));
        print_board(&life_board, &rows, &cols);
        for (r0, rl, c0, cl) in &extents {
            capture_moves(&life_board,
                          &rows,
                          &cols,
                          &r0,
                          &rl,
                          &c0,
                          &cl,
                          &mut moves);

            // handles.push(
            //     thread::spawn(|| {
            //         capture_moves(&life_board,
            //                       &rows,
            //                       &cols,
            //               &r0,
            //               &rl,
            //               &c0,
            //               &cl,
            //                       &mut moves); }
            //     )
            // );
        }
        // while handles.len() > 0 {
        //     let h = handles.pop().unwrap();
        //     h.join().unwrap();
        // }

        while moves.len() > 0 {
            let (r, c, v) = moves.pop().unwrap();
            life_board[[r, c]] = v;
        }
    }

    // flip_board(&mut boards.get_mut(&1).unwrap().borrow_mut(), &rows, &cols);
    //
    // println!("{} {}", rows, cols);
    // println!("{:?}", boards.get(&0).unwrap());
    // println!("{:?}", boards.get(&1).unwrap());
    //
    // for i in 0..3 {
    //     for j in 0..3 {
    //         println!("{}", count_neighbors(&boards.get(&0).unwrap().borrow(), &rows, &cols, &i, &j));
    //     }
    // }
}


// if old_board[[r, c]] == 1 {
//     if (count < 2) | (count > 3) {
//         new_board[[r, c]] = 0;
//         println!("{} {} {} {}", "dead", r, c, count);
//     }
// }
// else if count == 3 {
//     new_board[[r, c]] = 1;
//     println!("{} {} {} {}", "alive", r, c, count);
// }
// else {
//     new_board[[r, c]] = old_board[[r, c]];
// }