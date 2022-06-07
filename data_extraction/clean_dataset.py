import re

import pandas as pd

from common import read_config

MODULE_NAME = "clean_dataset"


def remove_rows_with_missing_data(df):
    """
    Removes any row containing nan or empty string values in the dataframe
    """
    df = df.drop(df[df.isna().any(axis=1)].index)
    df = df.drop(df[(df == "").any(axis=1)].index)
    return df


def clean_text_formatting(df):
    """
    Strips text of whitespaces
    """
    df.title = df.title.str.strip()
    df.body = df.body.str.strip()
    df.target = df.target.str.strip()
    return df


def remove_abusive_language_data(df, abusive_df_path):
    """
    Removes data from a manually annotated list of samples
    containing labels with abusive language
    """
    abusive_df = pd.read_csv(abusive_df_path, sep="\t", index_col=0)
    # FIXME: is errors="ignore" needed?
    df = df.drop(abusive_df[abusive_df["OK"] != "y"].index, errors="ignore")
    return df


def remove_clicks_saved(df):
    """
    Removes labels describing number of clicks saved.
    TODO: should n > 1 clicks saved be removed all together?
    Reason: most likely a js slideshow
    """
    
    pattern_repl = [
        # this pattern replaces brackets containing 'click' with surrounding spaces with a space
        (r"( \[[^\[]*click[^\]]*\] | \([^\(]*click[^\)]*\) )", " "),
        # this pattern replaces removes brackets containing 'click' with a leading or trailing space
        (r"( \[[^\[]*click[^\]]*\]|\[[^\[]*click[^\]]*\] | \([^\(]*click[^\)]*\)|\([^\(]*click[^\)]*\) )", ""),
        (r"([0-9]+\+? )?(saved.*clicks?|clicks?.*saved)", ""),
        (r"[0-9]+\+? clicks", ""),
    ]

    compiled_pattern_repl = [(re.compile(pattern, re.IGNORECASE), repl) for pattern, repl in pattern_repl]
    
    for pattern, repl in compiled_pattern_repl:
        df.target = df.target.str.replace(pattern, repl)
    
    return df


def remove_list_in_comments(df):
    """
    Removes rows that contain "list in comments" in the label. Note that the dataset could be expanded to support "list in comments" in the future.  
    """
    pattern = re.compile(r"(?=.*list)(?=.*comments)", re.IGNORECASE)
    results = df[df.target.str.contains(pattern)]
    df = df.drop(results.index)
    return df
    

def remove_too_short_documents(df, min_len_chars):
    df = df.drop(df[df.body.map(len) < min_len_chars].index)
    return df


def remove_too_long_documents(df, max_len_chars):
    """
    Some documents are too long due to parsing errors. 
    A small minority of the documents are very long, so we decide to remove the longest documents
    """
    df = df.drop(df[df.body.map(len) >= max_len_chars].index)
    return df


def clean_dataset(df, config):
    """
    Returns a clean dataset
    """
    df = remove_rows_with_missing_data(df)
    df = clean_text_formatting(df)
    df = df.astype("string")

    df = df.drop_duplicates("body")
    df = remove_too_short_documents(df, 100)
    df = remove_too_long_documents(df, 50000)
    df = remove_abusive_language_data(df, config["profanity_annotations_path"])
    df = remove_clicks_saved(df)
    df = remove_list_in_comments(df)
    
    df = clean_text_formatting(df)
    df = remove_rows_with_missing_data(df)

    return df


def main():
    config = read_config([MODULE_NAME, "datapaths"])
    module_config = config[MODULE_NAME]

    df_dirty = pd.read_csv(module_config["dirty_dataset_path"], sep="\t", index_col=0)
    df_clean = clean_dataset(df_dirty, module_config)

    assert df_clean.isna().any(axis=1).any() == False

    print("Removed ", len(df_dirty) - len(df_clean), "rows from dataset")

    df_clean.to_csv(module_config["clean_dataset_path"], sep="\t")
    

if __name__ == "__main__":
    main()