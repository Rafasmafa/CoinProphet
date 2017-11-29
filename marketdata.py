import os
import numpy as np
import pandas as pd
import pickle
import quandl
import csv



class ExchangeData(object):

    def __init__(self, coin):
        self.coin_name = coin
        self._coin_data = pd.Series([]) # create empty series

    @property
    def coin_data(self):
        if self._coin_data.empty:
            exchanges = ['COINBASE','BITSTAMP','ITBIT', 'KRAKEN']
            exchange_data = {}
            for exchange in exchanges:
                exchange_code = 'BCHARTS/{}USD'.format(exchange)
                exchange_df = self.get_quandl_data(exchange_code)
                exchange_data[exchange] = exchange_df
            data = self.merge_dfs_on_column(list(exchange_data.values()),
                                                 list(exchange_data.keys()),
                                                 'Weighted Price')
            data.replace(0, np.nan, inplace=True)
            data['avg_btc_price_usd'] = data.mean(axis=1)
            self._coin_data = data['avg_btc_price_usd']

        return self._coin_data

    @property
    def csv(self):
        return self.to_csv(self.coin_name, self.coin_data.index, self.coin_data)
 
    def merge_dfs_on_column(self, dataframes, labels, col):
        '''Merge a single column of each dataframe into a new combined dataframe'''
        series_dict = {}
        for index in range(len(dataframes)):
            series_dict[labels[index]] = dataframes[index][col]
        return pd.DataFrame(series_dict)

    def get_quandl_data(self, quandl_id):
        '''Download and cache Quandl dataseries'''
        cache_path = '{}.pkl'.format(quandl_id).replace('/','-')
        try:
            f = open(cache_path, 'rb')
            df = pickle.load(f)
            print('Loaded {} from cache'.format(quandl_id))
        except (OSError, IOError) as e:
            print('Downloading {} from Quandl'.format(quandl_id))
            df = quandl.get(quandl_id, returns="pandas")
            df.to_pickle(cache_path)
            print('Cached {} at {}'.format(quandl_id, cache_path))
        return df

    def to_csv(self, fn, ds, y):
        fn = fn+'.csv'
        with open(fn, 'w', newline='') as filename:
            writer = csv.writer(filename, delimiter=',',
                                 quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['"ds"', '"y"'])
            for i in range(0, len(ds)):
                writer.writerow([ds[i], y[i]])
        return os.path.abspath(fn)


