import os
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from common import read_config

MODULE_NAME = "split_dataset"


def read_ids(path: str) -> Tuple[List[str]]:
    """
    Reads ids from the specified path
    """
    with open(path, "r") as f:
        train_ids, val_ids, test_ids = map(
            lambda x: x.strip("\n").split(","), f.readlines()
        )

    return train_ids, val_ids, test_ids


def write_ids(
    path: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    """
    Writes ids to the specified path
    """
    with open(path, "w+") as f:
        dataframes = (train_df, val_df, test_df)
        f.writelines(",".join(i.index) + "\n" for i in dataframes)


def split_df(df: pd.DataFrame) -> Tuple[pd.DataFrame]:
    """
    Splits the dataset dataframe to a train, val and test dataset
    """
    train_set, test_val_set = train_test_split(df, test_size=0.2, random_state=42)

    test_size = len(test_val_set) // 2

    val_set = test_val_set[:test_size]
    test_set = test_val_set[test_size:]

    return train_set, val_set, test_set


def load_dfs_from_ids(
    df: pd.DataFrame, train_ids: List[str], val_ids: List[str], test_ids: List[str]
) -> Tuple[pd.DataFrame]:
    """
    Extrait train-val-test splits from lists of ids
    """

    train_set = df.loc[train_ids]
    val_set = df.loc[val_ids]
    test_set = df.loc[test_ids]

    return train_set, val_set, test_set


def handle_missing_id_path(df: pd.DataFrame, local_path: str) -> List[str]:
    """
    Called if the common
    """
    print(
        f"WARNING: ID FILE PATH NOT FOUND. "
        "THIS MEANS YOU MIGHT GET A DIFFERENT TRAIN, VAL AND TEST SET!"
        "Id file: "
    )
    proceed = False
    while not proceed:
        print("Proceed to make new local ids? [Y/n]")
        response = input(">")
        if response in "Yy":
            proceed = True
        elif response in "Nn":
            exit()
    print(f"Writing to {local_path}")
    splits = split_df(df)
    write_ids(local_path, *splits)
    ids = read_ids(local_path)
    return ids


def split_dataset(
    dataset_path: str,
    expected_datasplit_id_path: str,
    backup_datasplit_id_path: str,
    split_paths: Dict[str, str],
) -> None:
    """
    Splits the dataset to a training set, validation set and test set
    """
    df = pd.read_csv(dataset_path, index_col=0, sep="\t")

    if os.path.exists(expected_datasplit_id_path):
        ids = read_ids(expected_datasplit_id_path)
    else:
        ids = handle_missing_id_path(df, backup_datasplit_id_path)

    train_set, val_set, test_set = load_dfs_from_ids(df, *ids)

    train_set.to_csv(split_paths["train_path"], sep="\t")
    val_set.to_csv(split_paths["val_path"], sep="\t")
    test_set.to_csv(split_paths["test_path"], sep="\t")


def main() -> None:
    config = read_config([MODULE_NAME, "datapaths"])
    module_config = config[MODULE_NAME]

    split_dataset(
        dataset_path=config["datapaths"]["dataset_path"],
        expected_datasplit_id_path=module_config["train_val_test_id_path"],
        backup_datasplit_id_path=module_config["train_val_test_id_local_path"],
        split_paths=module_config,
    )


if __name__ == "__main__":
    main()
