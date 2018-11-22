import warnings
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from pppredictor import PPPredictor


def get_logger(nm, filename):
    lgr = logging.getLogger(nm)
    lgr.setLevel(logging.INFO)

    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    lgr.addHandler(handler)
    return lgr


def mean_absolute_percentage_error(y_true, y_pred):
    return abs((y_true - y_pred) / y_true) * 100


def validate(data, date):
    ppp = PPPredictor()
    prepared_prices_df, prepared_phrases_df, prepared_news_df = ppp.prepare_data(data, date)
    ppp.fit((prepared_prices_df, prepared_phrases_df, prepared_news_df), use_text_model=True)

    future_date = pd.to_datetime(date) + timedelta(days=91)

    # shift to monday
    nearest_monday = future_date
    while nearest_monday not in prepared_prices_df.index:
        nearest_monday -= timedelta(days=1)

    test_y = prepared_prices_df.at[nearest_monday, 'PPSpotAvgPrice']
    predicted = ppp.predict(future_date)
    return mean_absolute_percentage_error(test_y, predicted)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    logger = get_logger(__name__, 'better_logs.txt')

    start_date = '2015-12-01'
    end_date = '2016-06-30'

    prices_df = pd.read_csv('data/retrieved_data.csv', index_col='Date')
    phrases_df = pd.read_csv('data/proposed_phrases.csv')
    news_df = pd.read_csv('data/proposed_sites.csv', index_col='Date')

    scores = []

    for dt in pd.date_range(start=pd.to_datetime(start_date),
                            end=pd.to_datetime(end_date), freq='M'):
        str_dt = datetime.strftime(dt, '%Y-%m-%d')
        score = validate((prices_df, phrases_df, news_df), str_dt)
        scores.append(score)
        logger.info('MAPE for {}: {:.2f}%'.format(str_dt, score))
    logger.info('\nmean score: {:.2f}%'.format(np.mean(scores)))
    logger.info('---------------------------------------------')
