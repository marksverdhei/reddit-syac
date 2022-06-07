from typing import Dict, List

import pandas as pd
import toml

CONFIG_PATH = "config.toml"


def read_config(fields: List[str]) -> Dict[str, str]:
    """
    Loads a subset of config.toml with the supplied fields
    """
    conf = toml.load(CONFIG_PATH)
    subconf = {k: conf[k] for k in fields}
    return subconf


def read_tsv(path):
    """
    Simple utility function that standardizes the 
    arguments for the tsv files of the dataset
    """
    return pd.read_csv(path, sep="\t", index_col=0)


def update_preprocessing_log(data_size, task_name) -> None:
    """
    A function to keep track of the lost data in each data preprocessing step
    """
    preprocessing_log = pd.read_csv("../data/intermediate/preprocessing_log.csv", index_col=0)
    preprocessing_log.loc[task_name]["Samples"] = data_size
    preprocessing_log.to_csv("../data/intermediate/preprocessing_log.csv")
