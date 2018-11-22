import pandas as pd

from pppredictor import PPPredictor

date = '2017-08-07'
predict_date = '2017-11-06'

prices_df = pd.read_csv('data/retrieved_data.csv', index_col='Date')
phrases_df = pd.read_csv('data/proposed_phrases.csv')
news_df = pd.read_csv('data/proposed_sites.csv', index_col='Date')

ppp = PPPredictor()
prepared_prices_df, prepared_phrases_df, prepared_news_df = ppp.prepare_data((prices_df, phrases_df, news_df), date)
ppp.fit((prepared_prices_df, prepared_phrases_df, prepared_news_df), use_text_model=True)
predicted_price = ppp.predict(predict_date)
print('price:', predicted_price)
