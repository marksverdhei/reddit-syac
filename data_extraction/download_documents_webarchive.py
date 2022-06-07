import logging
import os
import pickle
from typing import Dict

import pandas as pd
import requests
from requests import Response
from tqdm import tqdm

from common import update_preprocessing_log

TASK_NAME = "download_documents"
TOO_MANY_REQUESTS_TIMEOUT = 12


def fetch_responses(data: pd.DataFrame) -> Dict[str, Response]:
    """
    Uses requests to retrieve responses
    """

    responses = {}
    tmr_counter = 0

    try:
        for id in tqdm(data.index):
            url = data["url"].loc[id]
            logging.info(url)

            try:
                response = requests.get(url)
                logging.info(response)

                if response.status_code == 429:
                    tmr_counter += 1
                else:
                    tmr_counter = 0

                if tmr_counter >= TOO_MANY_REQUESTS_TIMEOUT:
                    raise TimeoutError("You're busted! 429 12 times in a row!")
            except (
                requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
            ) as e:
                logging.warning(e)
                continue

            responses[id] = response
    except Exception as e:
        logging.error(e)
    finally:
        return responses


def main():
    if not os.path.exists("dataset_raw"):
        os.mkdir("dataset_raw")

    data = pd.read_csv("../data/intermediate/webarchive_urls.tsv", index_col=0, sep="\t")

    downloaded = [i[:-5] for i in os.listdir("dataset_raw/")]
    data.drop(downloaded, inplace=True, axis=0, errors="ignore")

    responses = fetch_responses(data)

    for id, response in tqdm(responses.items()):
        if response.status_code == 200:
            with open(f"dataset_raw/{id}.html", "wb+") as f:
                f.write(response.content)

    with open("../data/intermediate/requests_responses.pickle", "wb+") as f:
        pickle.dump(responses, f)

    update_preprocessing_log(len(os.listdir("dataset_raw")), TASK_NAME)


if __name__ == "__main__":
    main()
