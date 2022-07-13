import logging
import os
import pickle
from typing import Dict

import pandas as pd
import requests
from requests import Response
from tqdm import tqdm

TOO_MANY_REQUESTS_TIMEOUT = 12


def get_webarchive_df():
    """
    Returns the subset of the reddit syac dataset where the URL contains web.archive
    """
    splits = ("train", "validation", "test")
    df = pd.concat((pd.read_csv(f"data/public/{s}_urls.csv", index_col=0) for s in splits))
    return df[df.url.map(lambda url: "web.archive" in url)]


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
    html_dir = "data/html"
    web_archive_dir = f"{html_dir}/web_archive"

    for path in html_dir, web_archive_dir:
        if not os.path.exists(path):
            os.mkdir(path)

    data = get_webarchive_df()

    downloaded = [i[:-5] for i in os.listdir(web_archive_dir)]
    data.drop(downloaded, inplace=True, axis=0, errors="ignore")

    responses = fetch_responses(data)

    for id, response in tqdm(responses.items()):
        if response.status_code == 200:
            with open(f"{web_archive_dir}/{id}.html", "wb+") as f:
                f.write(response.content)

    with open("data/intermediate/requests_responses.pickle", "wb+") as f:
        pickle.dump(responses, f)


if __name__ == "__main__":
    main()
