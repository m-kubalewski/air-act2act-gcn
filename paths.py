import os

# Path to the main directory
main_dir = os.getcwd()

# Path to the directory with raw, unprocessed data
raw_data_dir_name = "raw_data"
raw_data_dir = os.path.join(main_dir, raw_data_dir_name)

# Path to the directory with processed .npz files, which contain adjacency matrix, node features, edge features and labels
npz_data_dir_name = "npz_data"
npz_data_dir = os.path.join(main_dir, npz_data_dir_name)
