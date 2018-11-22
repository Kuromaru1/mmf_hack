import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

f = open("oil_links.html", 'r', encoding='utf-8')

db = pd.DataFrame(columns=['SiteName', 'Header', 'Text', 'Date'])

t = f.read()
# print(t)
oil_soup = BeautifulSoup(t, "html.parser")
search_titles = oil_soup.find_all("span", "s2")
links = []
for title in search_titles:
    try:
        links += [title.a['href']]
    except:
        continue
print(len(links))
i = 0

for link in links:
    try:
        text = requests.get(link).text
    except:
        print("oh no")
        continue
    # print(text)
    soup = BeautifulSoup(text, "html.parser")
    # print(soup.prettify())
    # print(soup)
    try:
        texts = soup.findAll("div", "StandardArticleBody_body")[0].text
        header = soup.findAll("h1", "ArticleHeader_headline")[0].text
    except:
        print("daym")
        continue
    if i % 50 == 0:
        db.to_csv("retrieved_articles_oil.csv")
    # print(texts)
    t = soup.find("div", "ArticleHeader_date").text
    raw_date = t[:t.find('/')]
    date = datetime.strptime(raw_date, "%B %d, %Y ")
    article_date = datetime.strftime(date, "%d-%m-%Y")
    print(article_date)
    db = db.append({'SiteName': "https://www.reuters.com", 'Header': header, 'Text': texts, 'Date': article_date}, ignore_index=True)
    i += 1