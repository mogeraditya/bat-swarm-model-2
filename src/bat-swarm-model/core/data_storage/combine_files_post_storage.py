import glob
import gzip
import pickle

import h5py
import numpy as np
import pandas as pd
from supporting_files.utilities import load_parameters, make_dir
from tqdm.rich import tqdm


def stitch_together_pickle_history_lists_and_store(
    history_output_dir, dir_to_store, id_for_simulation
):
    """Merges lists from all the pickle files together.

    Args:
        history_output_dir (string):

    Returns:
        list: list with all the merged lists from pickle files.
    """

    list_of_dict_files = glob.glob(history_output_dir + "/history_dump_*.pkl.gz")
    list_of_dict_files = np.sort(list_of_dict_files)

    list_containing_data_from_all_pickle_files = []
    for pickle_file in tqdm(list_of_dict_files, desc="reading dict files "):
        with gzip.open(pickle_file, "rb") as f:
            _list_containing_subset = pickle.load(f)
            list_containing_data_from_all_pickle_files.extend(_list_containing_subset)
    print(id_for_simulation)
    parameter_file = glob.glob(history_output_dir + "/parameters_used.json")[0]
    parameter_df = load_parameters(parameter_file)
    print(parameter_df)
    times = [i["time"] for i in list_containing_data_from_all_pickle_files]
    sorting_indices = np.argsort(times)
    list_containing_data_from_all_pickle_files = np.array(
        list_containing_data_from_all_pickle_files
    )
    list_containing_data_from_all_pickle_files = (
        list_containing_data_from_all_pickle_files[sorting_indices]
    )
    # list_containing_data_from_all_pickle_files.append(parameter_df)
    np.append(list_containing_data_from_all_pickle_files, parameter_df)
    make_dir(dir_to_store)
    with gzip.open(dir_to_store + f"/history_{id_for_simulation}.pkl.gz", "wb") as f:
        pickle.dump(list_containing_data_from_all_pickle_files, f)
    return list_containing_data_from_all_pickle_files


def merge_summary_files(results_dir, output_file=None):
    """Merges the summary_dataframe csv of every job folder into one csv,
    keeping a single header (pd.concat as in
    https://stackoverflow.com/a/30335657).

    Args:
        results_dir (string): directory containing one folder per job.
        output_file (string, optional): path of the merged csv. Defaults to
            results_dir/summary_all_jobs.csv.

    Returns:
        DataFrame: merged summary, one row per job.
    """
    if output_file is None:
        output_file = results_dir + f"/summary_all_jobs.csv"

    summary_files = glob.glob(results_dir + "/*/*/*/summary_dataframe_job_*.csv")
    # print(summary_files)
    merged_df = pd.concat(
        (
            pd.read_csv(summary_file, index_col=0)
            for summary_file in tqdm(summary_files)
        ),
        ignore_index=True,
    )
    merged_df = merged_df.sort_values("job id", ignore_index=True)
    merged_df.to_csv(output_file, index=False)

    print(f"merged {len(summary_files)} summary files into {output_file}")
    return merged_df
