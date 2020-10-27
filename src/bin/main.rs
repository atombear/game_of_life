use ndarray_csv::Array2Reader;
use std::{thread::sleep, time::Duration, thread, sync::RwLock, sync::Arc, sync::mpsc};
use std::thread::JoinHandle;

use game_of_life::utils;

fn main() {
    // Get the current path, go up 3 directories, then find the board.
    let mut csv_path: std::path::PathBuf = std::env::current_exe().unwrap();
    for _ in 0..3 { csv_path.pop(); }
    for val in vec!["src", "board.csv"] { csv_path.push(val); }

    // Build the csv reader.
    let mut reader = csv::ReaderBuilder::new().has_headers(false)
                                              .from_path(csv_path)
                                              .expect("Cannot read file");
    // Use the csv reader to obtain the starting board.
    let starting_board: ndarray::Array2<u8> = reader.deserialize_array2_dynamic().unwrap();
    // Prepare the data for parallel read operations.
    // to_owned returns a copy.
    // RwLock allows multiple simultaneous read access, and single write access.
    // 'Automatically Reference Counted' allows for multiple parallel references to exist.
    let data_board: Arc<RwLock<ndarray::Array2<u8>>> = Arc::new(RwLock::new(starting_board.to_owned()));

    // The shape of the board.
    let rows: usize = starting_board.shape()[0];
    let cols: usize = starting_board.shape()[1];

    // Create sub-grids.
    let extents: Vec<(usize, usize, usize, usize)> = utils::get_subgrids(rows, cols);

    // Create an array of thread handles.
    let mut handles: Vec<thread::JoinHandle<_>> = vec![];
    let mut h: JoinHandle<()>;

    for iter in 0..50 {

        let (tx, rx): (mpsc::Sender<(usize, usize, u8)>,
                       mpsc::Receiver<(usize, usize, u8)>) = mpsc::channel();

        // Clear screen.
        print!("{}[2J", 27 as char);
        sleep(Duration::from_millis(100 as u64));
        { utils::print_board(&(data_board.read().unwrap()), &rows, &cols, &iter); }

        for &(r0, rl, c0, cl) in &extents {
            let par_data = data_board.clone();
            let par_tx = (&tx).clone();
            handles.push(
                thread::spawn(move || {
                    for mv in utils::capture_moves(&(par_data.read().unwrap()),
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
            h = handles.pop().unwrap();
            h.join().unwrap();
        }

        drop(tx);

        for mv in &rx {
            let (r, c, v) = mv;
            { data_board.write().unwrap()[[r, c]] = v; }
        }
    }
}


