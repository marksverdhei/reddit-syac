import logging
from collections import Counter
from typing import Tuple

import pandas as pd

from common import update_preprocessing_log

TASK_NAME = "filter_urls"

# Rules for dataset:
# No websites with less than 100 instances
# No video or image websites or meta posts
BLACKLIST_DOMAINS = [
    "streamable.com",  # video
    "youtube.com",  # video
    "i.redd.it",  # image
    "reddit.com",  # meta
    "unv.is",  # doesnt work for requests
    "unvis.it",  # doesnt work for requests
]


def get_domain(url: str) -> str:
    return (
        url.replace("https://", "")
        .replace("http://", "")
        .replace("www.", "")
        .split("/")[0]
    )


def get_domains(df: pd.DataFrame) -> pd.DataFrame:
    urls = df["url"]
    urls = urls.map(get_domain)
    return urls


def filter_invalid_domains(df_raw: pd.DataFrame) -> pd.DataFrame:
    domains = get_domains(df_raw)
    counts = Counter(domains)
    frequent_domains = domains[domains.apply(lambda s: counts[s] >= 100)]
    valid_domains = frequent_domains[
        frequent_domains.apply(lambda s: s not in BLACKLIST_DOMAINS)
    ]
    logging.info(valid_domains.unique())
    new_df = df_raw.loc[valid_domains.index]
    return new_df


def separate_archive_today_urls(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    is_archive = get_domains(df).map(lambda s: s.startswith("archive"))
    archive_df = df[is_archive]
    non_archive_df = df[~is_archive]
    return archive_df, non_archive_df


def main() -> None:
    df_raw = pd.read_csv("../data/intermediate/syac_urls_raw.tsv", index_col=0, sep="\t")
    new_df = filter_invalid_domains(df_raw)

    archive_df, non_archive_df = separate_archive_today_urls(new_df)

    archive_df.to_csv("../data/intermediate/archivetoday_urls.tsv", sep="\t")
    non_archive_df.to_csv("../data/intermediate/webarchive_urls.tsv", sep="\t")

    update_preprocessing_log(len(archive_df) + len(non_archive_df), TASK_NAME)


if __name__ == "__main__":
    main()
