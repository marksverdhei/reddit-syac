import os

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm
from time import sleep

chrome_options = Options()
# This line disables javascript
chrome_options.add_experimental_option("prefs", {'profile.managed_default_content_settings.javascript': 2})

DATA_PATH = "dataset_raw/archive_today"
captcha_content = "Completing the CAPTCHA proves you are a human"


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
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    data = pd.read_csv("../data/intermediate/archivetoday_urls.tsv", index_col=0, sep="\t")
    local_doc_ids = get_downloaded_document_ids()
    data.drop(local_doc_ids, inplace=True, errors="ignore")
    download_pages(data)


if __name__ == "__main__":
    main()