import pandas as pd
import praw
import dotenv

CREDS = dotenv.dotenv_values(".env")
session = praw.Reddit(**CREDS)


archive_today_urls = pd.read_csv("../data/intermediate/archivetoday_urls.tsv", sep="\t", index_col=0)
web_archive_urls = pd.read_csv("../data/intermediate/webarchive_urls.tsv", sep="\t", index_col=0)

full_url_df = pd.concat((archive_today_urls, web_archive_urls))
# requests_urls_raw = pd.read_csv("../../data/intermediate/url_data_raw.tsv", sep="\t", index_col=0)
# full_url_df = requests_urls_raw

for s in "train", "validation", "test":
    df = pd.read_csv("data/" + s + ".csv", index_col=0)

    id_diff = set(df.index) - set(full_url_df.index)
    df["url"] = full_url_df["url"]


    if id_diff:
        remaining_urls = {i: session.submission(id=i).url for i in id_diff}
        for i, url in remaining_urls.items():
            print(i)
            df.loc[i]["url"] = url

    print(df.isna().any())
    df[["title", "url", "target"]].to_csv(s + "_urls.csv")

# train = pd.read_csv("train.csv", index_col=0)
# validation = pd.read_csv("validation.csv", index_col=0)
# test = pd.read_csv("test.csv", index_col=0)

