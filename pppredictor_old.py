import pandas as pd
import numpy as np
from datetime import timedelta

from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, Embedding


class PPPredictor:

    def __init__(self):
        self.lstm = None
        self.lstm_train_X = None
        self.train_y = None
        self.embedding = None
        self.embedding_train_X = None
        self.week_shift = 13
        self.week_news_expiration = 4
        self.predict_with_embeddings = False
        self.nn_coef = (1., 0.)

    def __build_lstm(self, df_shape):
        model = Sequential()

        model.add(LSTM(input_dim=df_shape[1], output_dim=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(output_dim=1))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer='rmsprop')
        return model

    def __build_embedding(self):
        max_features = 20000

        model = Sequential()
        model.add(Embedding(max_features, 128))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def __get_lstm_train(self, df):
        self.scaler = MinMaxScaler()
        train_X = df.drop(['PPSpotAvgPrice'], axis=1)
        train_X['Date'] = pd.to_numeric(train_X.index)
        train_X = self.scaler.fit_transform(train_X)
        train_X = train_X[:-self.week_shift]
        train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))

        self.scaler_y = MinMaxScaler()
        train_y = np.reshape(np.array(df['PPSpotAvgPrice']), (-1, 1))
        train_y = self.scaler_y.fit_transform(train_y)
        train_y = train_y[self.week_shift:, 0]

        return train_X, train_y

    def __concat_texts_with_prices(self, texts, prices):
        prices = prices[self.week_shift:]
        df = pd.DataFrame({prices.name: prices, texts.name: ""}, index=prices.index)
        for i, row in df.iterrows():
            lower_week_bound = i - timedelta(days=7 * self.week_news_expiration)
            df.at[i, texts.name] = ' '.join(
                texts[(texts.index >= lower_week_bound) & (texts.index <= i)].values.tolist())
        return df

    def __get_embedding_train(self, df, labels):
        df = self.__concat_texts_with_prices(df['Text'], labels)
        tokenizer = Tokenizer(nb_words=5000)
        tokenizer.fit_on_texts(df['Text'])
        sequences = tokenizer.texts_to_sequences(df['Text'])

        # word_index = tokenizer.word_index
        # print('Found %s unique tokens.' % len(word_index))

        train_X = pad_sequences(sequences, maxlen=80)
        return train_X

    def prepare_data(self, data, actual_date):
        prices_df, phrases_df, news_df = data[0].copy(), data[1].copy(), data[2].copy()

        prices_df.index = pd.to_datetime(prices_df.index)
        for col in prices_df.columns:
            prices_df[col].interpolate(method='time', inplace=True)
            prices_df[col].fillna(method='bfill', inplace=True)
        prices_df = prices_df[prices_df.index <= pd.to_datetime(actual_date)]

        news_df.index = pd.to_datetime(news_df.index)
        news_df.drop(['SiteName'], axis=1, inplace=True)
        news_df.sort_values(by='Date', inplace=True)
        news_df = news_df[news_df.index <= pd.to_datetime(actual_date)]

        return prices_df, phrases_df, news_df

    def fit(self, prepared_data, use_text_model=False):
        prices_df, phrases_df, news_df = prepared_data

        self.lstm = self.__build_lstm(prices_df.shape)
        self.lstm_train_X, self.train_y = self.__get_lstm_train(prices_df)
        # self.lstm.fit(self.lstm_train_X, self.train_y, batch_size=128, nb_epoch=10)

        if use_text_model:
            self.embedding = self.__build_embedding()
            self.embedding_train_X = self.__get_embedding_train(news_df, prices_df['PPSpotAvgPrice'])
            self.embedding.fit(self.embedding_train_X, self.train_y, batch_size=32, epochs=15)
        self.predict_with_embeddings = use_text_model

        return self.lstm

    def predict(self, date):
        # test_X = self.lstm_train_X[-1]
        # test_X = np.reshape(test_X, (1, 1, test_X.shape[1]))
        # forecast = self.lstm.predict(test_X)
        # return self.scaler_y.inverse_transform(forecast)[0][0]

        test_X = self.embedding_train_X[-1]
        forecast = self.embedding.predict(test_X)
        return self.scaler_y.inverse_transform(forecast)[0][0]
