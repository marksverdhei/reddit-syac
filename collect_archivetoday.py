import os

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm
from time import sleep

chrome_options = Options()
# This line disables javascript
chrome_options.add_experimental_option("prefs", {'profile.managed_default_content_settings.javascript': 2})

DATA_PATH = "data/html/archive_today"
captcha_content = "Completing the CAPTCHA proves you are a human"


def get_archivetoday_df():
    """
    Returns the subset of the reddit syac dataset where the URLs that do not contain web.archive
    """
    splits = ("train", "validation", "test")
    df = pd.concat((pd.read_csv(f"data/public/{s}_urls.csv", index_col=0) for s in splits))
    return df[df.url.map(lambda url: "web.archive" not in url)]


def get_downloaded_document_ids():
    "Removes extensions from html files: abc123.html -> abc123"
    return [i[:-5] for i in os.listdir(DATA_PATH)]


def is_captcha_blocked(document):
    return captcha_content in document


def download_pages(data):
    """
    Downloads pages using selenium web driver
    """

    n_retries_threshold = 5
    n_retries = 0

    while (len(data) > 0) and (n_retries < n_retries_threshold):
        driver = webdriver.Chrome('chromedriver', chrome_options=chrome_options)
        
        for i in tqdm(data.index):
            try:
                url = data.loc[i]["url"]
                
                driver.get(url)
                content = driver.page_source

                assert not is_captcha_blocked(content)
                
                with open(f"{DATA_PATH}/{i}.html", "w+", encoding="UTF8") as f:
                    f.write(content)

                # We delay each request with
                # 1 second to not overload the site
                sleep(1)

            except Exception as e:
                print("Error occurred:")
                print(str(e)[:500])
                with open("error.log", "a") as f:
                    f.write(str(e))

                print(f"Skipping {i} for now...")
                driver.quit()
                driver = webdriver.Chrome('chromedriver', chrome_options=chrome_options)

        local_doc_ids = get_downloaded_document_ids()
        data.drop(local_doc_ids, inplace=True, errors="ignore")

        n_retries += 1
        driver.quit()


def main():
    html_dir = "data/html"
    archive_today_dir = f"{html_dir}/archive_today"

    for path in html_dir, archive_today_dir:
        if not os.path.exists(path):
            os.mkdir(path)


    data = get_archivetoday_df()
    local_doc_ids = get_downloaded_document_ids()
    data.drop(local_doc_ids, inplace=True, errors="ignore")
    download_pages(data)


if __name__ == "__main__":
    main()