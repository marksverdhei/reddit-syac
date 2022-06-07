# Reddit SYAC dataset  

Reddit SYAC (Saved you a click) is an abstractive title answering dataset consisting in total of 8608 examples.  

This repository contains all the code used to fetch, extract and assemble the Reddit SYAC dataset.  

## Public variant  

The public dataset variant consists of reddit post ids, titles, and urls to archived pages for clickbait articles.  
The two archiving pages used are web.archive and archive.today variants. These pages require different methods for scraping, 
so we have one script for each domain.  

## Private variant  

After running the scripts to collect, extract, assemble and split the dataset, you should have a train set of 7608 examples, a validation set of 500 examples, and a test set of 500 examples. The test set is manually curated, so it is important to ensure that you get the correct test.  
