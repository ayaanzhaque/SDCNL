# Web-Scraper for Reddit Data

# Data used for paper and results were last scraped in September 2020. 
# Adapted from (https://github.com/hesamuel/goodbye_world/blob/master/code/01_Data_Collection.ipynb

# data analysis imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# NLP Imports
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

from sklearn.feature_extraction.text import CountVectorizer

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import wordninja

# creating user agent
headers = {"User-agent" : "randomuser"} # set user agent to reddit account username
url_1 = "https://www.reddit.com/r/depression.json"

res = requests.get(url_1, headers=headers)
res.status_code

# scraper function
def reddit_scrape(url_string, number_of_scrapes, output_list):
    #scraped posts outputted as lists
    after = None 
    for _ in range(number_of_scrapes):
        if _ == 0:
            print("SCRAPING {}\n--------------------------------------------------".format(url_string))
            print("<<<SCRAPING COMMENCED>>>") 
            print("Downloading Batch {} of {}...".format(1, number_of_scrapes))
        elif (_+1) % 5 ==0:
            print("Downloading Batch {} of {}...".format((_ + 1), number_of_scrapes))
        
        if after == None:
            params = {}
        else:
            #THIS WILL TELL THE SCRAPER TO GET THE NEXT SET AFTER REDDIT'S after CODE
            params = {"after": after}             
        res = requests.get(url_string, params=params, headers=headers)
        if res.status_code == 200:
            the_json = res.json()
            output_list.extend(the_json["data"]["children"])
            after = the_json["data"]["after"]
        else:
            print(res.status_code)
            break
        time.sleep(randint(1,6))
    
    print("<<<SCRAPING COMPLETED>>>")
    print("Number of posts downloaded: {}".format(len(output_list)))
    print("Number of unique posts: {}".format(len(set([p["data"]["name"] for p in output_list]))))

# remove any repeat posts
def create_unique_list(original_scrape_list, new_list_name):
    data_name_list=[]
    for i in range(len(original_scrape_list)):
        if original_scrape_list[i]["data"]["name"] not in data_name_list:
            new_list_name.append(original_scrape_list[i]["data"])
            data_name_list.append(original_scrape_list[i]["data"]["name"])
    #CHECKING IF THE NEW LIST IS OF SAME LENGTH AS UNIQUE POSTS
    print("LIST NOW CONTAINS {} UNIQUE SCRAPED POSTS".format(len(new_list_name)))

# scraping suicide_watch data
suicide_data = []
reddit_scrape("https://www.reddit.com/r/SuicideWatch.json", 50, suicide_data)

suicide_data_unique = []
create_unique_list(suicide_data, suicide_data_unique)

# add suicide_watch to dataframe
suicide_watch = pd.DataFrame(suicide_data_unique)
suicide_watch["is_suicide"] = 1
suicide_watch.head()

# scraping suicide_watch data
depression_data = []
reddit_scrape("https://www.reddit.com/r/depression.json", 50, depression_data)

depression_data_unique = []
create_unique_list(depression_data, depression_data_unique)

# add suicide_watch to dataframe
depression = pd.DataFrame(depression_data_unique)
depression["is_suicide"] = 0
depression.head()

# saving data
suicide_watch.to_csv('suicide_watch.csv', index = False)
depression.to_csv('depression.csv', index = False)

# creating combined CSV
depression = pd.read_csv('depression.csv')
suicide_watch = pd.read_csv('suicide_watch.csv')

dep_columns = depression[["title", "selftext", "author",  "num_comments", "is_suicide","url"]]
sui_columns = suicide_watch[["title", "selftext", "author",  "num_comments", "is_suicide","url"]]

combined_data = pd.concat([dep_columns, sui_columns],axis=0, ignore_index=True)  
combined_data["selftext"].fillna("emptypost",inplace=True)
combined_data.head()
combined_data.isnull().sum()

# saving combined CSV
combined_data.to_csv('suicide_vs_depression.csv', index = False)
