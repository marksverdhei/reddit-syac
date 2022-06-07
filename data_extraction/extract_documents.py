import logging
import multiprocessing
import os
from typing import List
from xml.dom.minidom import Document

import newspaper
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import update_preprocessing_log

TASK_NAME = "extract_documents"

SYAC_DATASET_PATH = "../data/syac_dataset_raw.tsv"

WEB_ARCHIVE_TSV_PATH = "../data/intermediate/webarchive_urls.tsv"
ARCHIVE_TODAY_TSV_PATH = "../data/intermediate/archivetoday_urls.tsv"

WEB_ARCHIVE_RAW_PATH = "dataset_raw/webarchive/"
ARCHIVE_TODAY_RAW_PATH = "dataset_raw/archivetoday/"

# TODO: change
data_types = {
    "webarchive": {
        "tsv": WEB_ARCHIVE_TSV_PATH,
        "raw_dir": WEB_ARCHIVE_RAW_PATH,
    },
    "archivetoday": {
        "tsv": ARCHIVE_TODAY_TSV_PATH,
        "raw_dir": ARCHIVE_TODAY_RAW_PATH,
    },
}


def extract_documents(df: pd.DataFrame) -> List[str]:
    """
    Uses newspaper.fulltext to parse the html document and
    extract a string representation of the news articles
    """

    extracted_documents = []

    for doc_id in tqdm(list(df.index)):
        try:
            # TODO: change this to "rb"?
            with open(df["path"].loc[doc_id], "r", encoding="UTF8") as f:
                doc_str = f.read()
            fulltext = newspaper.fulltext(doc_str)
            extracted_documents.append(fulltext)
        except (AttributeError, UnicodeDecodeError) as e:
            logging.info(e)
            df.drop(doc_id, inplace=True)

    df["body"] = extracted_documents
    return df


def assemble_syac_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies postprocessing to get the final dataset dataframe
    * Removes the URL column
    * Adds a title column
    * Adds a target column with the text after the first pipe (|)
    * Drops everything after the second pipe as it is usually about number of clicks saved
    """
    df = df.copy()
    df = df[df["title"].apply(lambda x: "|" in x)]
    
    # Strip data from leading and trailing spaces
    df["title"] = [i.split("|")[0].strip() for i in df["title"]]
    df["target"] = [i.split("|")[1].strip() for i in df["title"]]

    df.drop("url", axis=1, inplace=True)
    df.drop("path", axis=1, inplace=True)

    return df


def get_raw_document_ids(path):
    document_ids = [i[:-5] for i in os.listdir(path)]
    return document_ids


def get_df_for_local_docuemnts(tsv_path, local_dir):
    data = pd.read_csv(tsv_path, index_col=0, sep="\t")
    document_ids = get_raw_document_ids(local_dir)
    df = data.loc[data.index.intersection(document_ids)]
    df["path"] = [f"{local_dir}/{i}.html" for i in df.index]
    return df


def main() -> None:
    used_datasources = ["archivetoday", "webarchive"]

    dfs = [get_df_for_local_docuemnts(data_types[i]["tsv"], data_types[i]["raw_dir"]) for i in used_datasources]

    df = pd.concat(dfs)

    n_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(n_cores) as pool:
        df_fragments = np.array_split(df, n_cores * 4)
        extracted_data = pool.map(extract_documents, tqdm(df_fragments))

    df = pd.concat(extracted_data)

    dataset = assemble_syac_dataset(df)
    dataset.to_csv(SYAC_DATASET_PATH, sep="\t")
    update_preprocessing_log(len(dataset), TASK_NAME)


if __name__ == "__main__":
    main()
