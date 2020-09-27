use itertools::iproduct;
use ndarray::array;
use ndarray_csv::Array2Reader;
use std::{thread::sleep, time::Duration, thread, sync::RwLock, sync::Arc, sync::mpsc};

/// Split up the board into non-overlapping sub-boards.
static NUM_ROW_GROUPS: u64 = 3;
static NUM_COL_GROUPS: u64 = 3;

/*
Any live cell with fewer than two live neighbours dies, as if by underpopulation.
Any live cell with two or three live neighbours lives on to the next generation.
Any live cell with more than three live neighbours dies, as if by overpopulation.
Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
*/

/// Flip every value in a board.
///
/// # Arguments
/// * `brd` - the board.
/// * `rows` - the number of rows.
/// * `cols` - the number of columns.
///
/// ```
/// let mut arr = array![[0, 1], [1, 0]];
/// let r = arr.shape()[0];
/// let c = arr.shape()[1];
/// flip_board(&mut arr, &r, &c);
/// assert_eq!(arr, array![[1, 0], [0, 1]]);
/// ```
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

/// Add all the values in a board at a given collection of indices.
///
/// # Arguments
/// `brd` - the board.
/// `pos_arr` - the array of positions to sum.
/// # Returns
/// The sum.
///
/// ```
/// let mut arr = array![[1, 2], [3, 4]];
/// assert_eq!(gather_board_values(&arr, &[(0, 0), (1, 1)]), 5);
/// ```
fn gather_board_values(brd: &ndarray::Array2<u8>, pos_arr: &[(usize, usize)]) -> u8 {
    let mut ret: u8 = 0;
    for (r0, c0) in pos_arr.iter() { ret += brd[[*r0, *c0]]; }
    return ret
}

/// Determine the sum of all the neighbors of a given cell.
///
/// # Arguments
/// `brd` - the board.
/// `rows` - the number of rows.
/// `cols` - the number of columns.
/// `r` - the row of the cell.
/// `c` - the column of the cell.
///
/// # Returns
/// The sum of all neighbors of a particular cell.
///
/// ```
/// let mut arr = array![[0, 1, 2],
///                      [3, 4, 5],
///                      [6, 7, 8]];
///
/// let r = arr.shape()[0];
/// let c = arr.shape()[1];
///
/// assert_eq!(count_neighbors(&arr, &r, &c, &0, &0), 8);
/// assert_eq!(count_neighbors(&arr, &r, &c, &0, &1), 14);
/// assert_eq!(count_neighbors(&arr, &r, &c, &0, &2), 10);
///
/// assert_eq!(count_neighbors(&arr, &r, &c, &1, &0), 18);
/// assert_eq!(count_neighbors(&arr, &r, &c, &1, &1), 32);
/// assert_eq!(count_neighbors(&arr, &r, &c, &1, &2), 22);
///
/// assert_eq!(count_neighbors(&arr, &r, &c, &2, &0), 14);
/// assert_eq!(count_neighbors(&arr, &r, &c, &2, &1), 26);
/// assert_eq!(count_neighbors(&arr, &r, &c, &2, &2), 16);
/// ```
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

/// Print a board.
///
/// # Arguments
/// `brd` - the board.
/// `rows` - the number of rows.
/// `cols` - the number of columns.
fn print_board(brd: &ndarray::Array2<u8>,
               rows: &usize,
               cols: &usize,
               frame_num: &usize) {
    println!("{} {}", "Frame", frame_num);
    for r in 0..*rows {
        for c in 0..*cols {
            print!("{} ", brd[[r, c]]);
        }
        print!("{}", "\n");
    }
}

/// Iterate through a rectangular sub-board and return an array of tuples each of which designates a
/// change to the original board.
///
/// # Arguments
/// `brd` - the board.
/// `rows` - the number of rows.
/// `cols` - the number of columns.
/// `start_row` - the row the block starts on.
/// `stop_row` - the row the block stops on.
/// `start_col` - the column the block starts on.
/// `stop_col` - the column the block stops on.
///
/// # Returns
/// An array of moves, specifying the value in a board position.
///
/// ```
/// let mut arr = array![[0, 1, 0],
///                      [1, 1, 1],
///                      [1, 0, 0]];
/// let r = arr.shape()[0];
/// let c = arr.shape()[1];
/// assert_eq!(capture_moves(&arr, &r, &c, &0, &2, &0, &1), vec![(0, 0, 1), (1, 0, 1)]);
/// ```
fn capture_moves(brd: &ndarray::Array2<u8>,
                 rows: &usize,
                 cols: &usize,
                 start_row: &usize,
                 stop_row: &usize,
                 start_col: &usize,
                 stop_col: &usize) -> Vec<(usize, usize, u8)> {
    let mut moves: Vec<(usize, usize, u8)> = vec![];
    let mut count: u8;
    for (r, c) in iproduct!(*start_row..*stop_row, *start_col..*stop_col) {
        count = count_neighbors(brd, rows, cols, &r, &c);
        // These are the rules of the game of life - determining whether a cell lives or dies by
        // considering its neighbors.
        if (brd[[r, c]] == 1) & ((count < 2) | (count > 3)) {
            moves.push((r, c, 0)); }
        else if count == 3 {
            moves.push((r, c, 1)); }
    }
    return moves;
}

/// Arithmetically separate a number into num_groups groups as evenly as possible.
///
/// # Arguments
/// `num` - the number to separate.
/// `num_groups` - the number of groups.
///
/// # Returns
/// A vector with the group sizes.
///
/// ```
/// assert_eq!(get_groups(11, 2), vec![6, 5]);
/// assert_eq!(get_groups(42, 1), vec![42]);
/// assert_eq!(get_groups(20, 6), vec![4, 4, 3, 3, 3, 3]);
/// ```
fn get_groups(num: u64, num_groups: u64) -> Vec<usize> {
    let mut len: usize;
    let mut ret: Vec<usize> = vec![];
    let mut rem: u64 = num - num_groups * (num / num_groups);

    for _ in 0..num_groups {
        ret.push((num / num_groups) as usize);
        if rem > 0 {
            len = ret.len();
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

    let data0: ndarray::Array2<u8> = reader.deserialize_array2_dynamic().unwrap();
    // Prepare the data for parallel read operations.
    let data_board: Arc<RwLock<ndarray::Array2<u8>>> = Arc::new(RwLock::new(data0.to_owned()));

    let rows: usize = data0.shape()[0];
    let cols: usize = data0.shape()[1];

    let row_groups: Vec<usize> = get_groups(rows as u64, NUM_ROW_GROUPS);
    let col_groups: Vec<usize> = get_groups(cols as u64, NUM_COL_GROUPS);

    let mut extents: Vec<(usize, usize, usize, usize)> = vec![];

    let mut row_first_idx: usize = 0;
    let mut col_first_idx: usize;
    for r_len in &row_groups {
        col_first_idx = 0;
        for c_len in &col_groups {
            extents.push((row_first_idx,
                          row_first_idx + r_len,
                          col_first_idx,
                          col_first_idx + c_len));
            col_first_idx += c_len;
        }
        row_first_idx += r_len;
    }

    let mut handles: Vec<thread::JoinHandle<_>> = vec![];

    for iter in 0..50 {

        let (tx, rx): (mpsc::Sender<(usize, usize, u8)>,
                       mpsc::Receiver<(usize, usize, u8)>) = mpsc::channel();

        // Clear screen.
        print!("{}[2J", 27 as char);
        sleep(Duration::from_millis(500 as u64));
        { print_board(&(data_board.read().unwrap()), &rows, &cols, &iter); }

        for &(r0, rl, c0, cl) in &extents {
            let par_data = data_board.clone();
            let par_tx = (&tx).clone();
            handles.push(
                thread::spawn(move || {
                    for mv in capture_moves(&(par_data.read().unwrap()),
                                            &rows,
                                            &cols,
                                            &r0,
                                            &rl,
                                            &c0,
                                            &cl) {
                        par_tx.send(mv).unwrap();
                    }
                })
            );
        }

        while handles.len() > 0 {
            let h = handles.pop().unwrap();
            h.join().unwrap();
        }

        drop(tx);

        for mv in &rx {
            let (r, c, v) = mv;
            { data_board.write().unwrap()[[r, c]] = v; }
        }
    }
}


