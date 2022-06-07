import logging
import os
from typing import Any, Dict

# create module containing this yourself
import dotenv
import pandas as pd
import praw
import psaw

CREDS = dotenv.dotenv_values(".env")

SUBREDDIT_NAME = "savedyouaclick"
N_CALLS = 1


def fetch_url_dataset(include_praw=False) -> Dict[str, Any]:
    """
    Downloads all titles, urls and post IDs of non-deleted savedyouaclick posts
    """
    session = praw.Reddit(**CREDS)
    post_dict = fetch_url_dataset_psaw(session)

    if include_praw:
        post_dict.update(fetch_url_dataset_praw(session, N_CALLS))

    return post_dict


def fetch_url_dataset_praw(reddit_session, n_calls=1) -> Dict[str, Any]:
    """
    Fetches urls using the praw API
    The praw API limits your requested posts to a thousand, so repeated calls are made to increase number of posts

    # TODO: implement sliding window over date
    """
    subreddit = reddit_session.subreddit(SUBREDDIT_NAME)

    post_dict = {}

    for i in range(n_calls):
        post_dict_len = len(post_dict)
        postit = subreddit.new(limit=None)
        post_dict.update((p.id, p) for p in postit if p.is_robot_indexable)
        logging.info(f" Increased number of posts by {len(post_dict) - post_dict_len}")

    return post_dict


def fetch_url_dataset_psaw(session) -> Dict[str, Any]:
    api = psaw.PushshiftAPI(session)
    data = {
        p.id: p
        for p in api.search_submissions(subreddit=SUBREDDIT_NAME)
        if p.is_robot_indexable
    }
    logging.info(f" Retreived {len(data)} posts")
    return data


def create_preprocessing_log(data_size) -> None:
    """
    Creates a dataframe describing how much data is discarded
    in the pre-processing
    """
    preprocessing_log = pd.DataFrame(
        {"Samples": [data_size, 0, 0, 0]},
        index=[
            "download_urls",
            "filter_urls",
            "download_documents",
            "extract_documents",
        ],
    )

    preprocessing_log.to_csv("../data/intermediate/preprocessing_log.csv")


def main():
    if not os.path.exists("../data/intermediate"):
        os.mkdir("../data/intermediate")

    post_dict = fetch_url_dataset()

    raw_dataset = pd.DataFrame(
        {
            "post_title": [p.title for p in post_dict.values()],
            "url": [p.url for p in post_dict.values()],
        },
        index=post_dict.keys(),
    )

    raw_dataset.to_csv("../data/intermediate/syac_urls_raw.csv")

    with open("posts.pickle", "wb+") as f:
        f.write()

    create_preprocessing_log(len(raw_dataset))


if __name__ == "__main__":
    main()
