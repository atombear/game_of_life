use ndarray_csv::{Array2Reader, Array2Writer};
use ndarray::array;
use std::{cell::RefCell, collections::HashMap, thread::sleep, time::Duration, thread, sync::RwLock, sync::Arc, sync::mpsc};

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
/// use ndarray::array;
///
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
/// use ndarray::array;
///
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
/// use ndarray::array;
///
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

fn capture_moves(brd: &ndarray::Array2<u8>,
                 rows: &usize,
                 cols: &usize,
                 start_row: &usize,
                 stop_row: &usize,
                 start_col: &usize,
                 stop_col: &usize) -> Vec<(usize, usize, u8)> {
    let mut moves: Vec<(usize, usize, u8)> = vec![];
    for r in *start_row..*stop_row {
        for c in *start_col..*stop_col {
            let count = count_neighbors(brd, rows, cols, &r, &c);
            if (brd[[r, c]] == 1) & ((count < 2) | (count > 3)) {
                moves.push((r, c, 0)); }
            else if count == 3 {
                moves.push((r, c, 1)); }
        }
    }
    return moves;
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

    let data0: ndarray::Array2<u8> = reader.deserialize_array2_dynamic().unwrap();
    let data1: Arc<RwLock<ndarray::Array2<u8>>> = Arc::new(RwLock::new(data0.to_owned()));

    let rows: usize = data0.shape()[0];
    let cols: usize = data0.shape()[1];

    // let mut boards: HashMap<u8, Arc<RwLock<ndarray::Array2<u8>>>> = HashMap::new();
    // boards.insert(0, RwLock::new(data0));
    // boards.insert(1, Arc::new(RwLock::new(data1)));

    // Track the changes that must be made to the next iteration.
    // A board for visualization and a board for modification.
    // let viz_board = boards.get(&0).unwrap().borrow();
    // let mut life_board_rwlock = RwLock::new(boards.get(&1).unwrap().borrow_mut());

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

    // let life_board_rwlock = boards.get_mut(&1).unwrap();
    let mut handles: Vec<thread::JoinHandle<_>> = vec![];

    // let mut tx_vec: Vec<Arc<mpsc::Sender<(usize, usize, u8)>>> = vec![];
    // for _ in 1..extents.len() {
    //     let new_tx = (&tx).clone();
    //     tx_vec.push(Arc::new(new_tx));
    // }
    // tx_vec.push(Arc::new(tx));

    for iter in 0..50 {

        let (tx, rx): (mpsc::Sender<(usize, usize, u8)>,
               mpsc::Receiver<(usize, usize, u8)>) = mpsc::channel();

        // Clear screen.
        print!("{}[2J", 27 as char);
        sleep(Duration::from_millis(500 as u64));
        { print_board(&(data1.read().unwrap()), &rows, &cols, &iter); }

        for &(r0, rl, c0, cl) in &extents {
            let par_data = data1.clone();
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
            { data1.write().unwrap()[[r, c]] = v; }
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