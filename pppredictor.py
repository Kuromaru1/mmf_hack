import pandas as pd





if __name__ == '__main__':
    df = pd.read_csv('./data/retrieved_data.csv', index_col='Date')
    df.index = pd.to_datetime(df.index)
    for col in df.columns:
        df[col].interpolate(method='time', inplace=True)
    print(df)

    #print(prepare_data(df, '2017-11-12').head())
