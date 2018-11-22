import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

f = open("news-links.txt", 'r')

db = pd.DataFrame(columns=['SiteName', 'Text', 'Date'])

links = f.readlines()

for link in links:
    text = requests.get(link[:-1]).text
    # print(text)
    soup = BeautifulSoup(text, "html.parser")
    # print(soup.prettify())
    # print(soup)
    texts = soup.findAll("div", "StandardArticleBody_body")[0].text
    print(texts)
    t = soup.find("div", "ArticleHeader_date").text
    raw_date = t[:t.find('/')]
    date = datetime.strptime(raw_date, "%B %d, %Y ")
    article_date = datetime.strftime(date, "%d-%m-%Y")
    print(article_date)
    db = db.append({'SiteName': "https://www.reuters.com", 'Text': texts, 'Date': article_date}, ignore_index=True)

db.to_csv("./data/retrieved_articles.csv")
