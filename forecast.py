import argparse
import os
import pandas as pd
import numpy as np
from fbprophet import Prophet
from utils.exceptions import UnsupportedFileType
import marketdata


SUPPORTED_FILE_TYPES = ['.csv']


def determine_file_type(fn):
    ext = os.path.splitext(fn)[1]
    if ext in SUPPORTED_FILE_TYPES:
        return ext
    else:
        raise UnsupportedFileType('File type is not supported')

def forecast(args):
    # TODO: make file extension parsing better
    # Not sure if other file types will be ok
    # This is just quick coding
    md = marketdata.ExchangeData('BTC')
    df = pd.read_csv(md.csv)

    
    df['y'] = np.log(df['y'])
    df.head()
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=args.periods)
    future.tail()
    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    fig1 = m.plot(forecast, ylabel='Price', xlabel='Time')
    fig1.savefig(md.coin_name + '.png')
    if args.plot_components:
        fig2 = m.plot_components(forecast)
        fig2.savefig(md.coin_name + '_components.png')
    print('done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',default=None, dest='data',
                        help='input data file. Supported file types: CSV')
    parser.add_argument('--growth', dest='growth', default=None,
                        help="String 'linear' or 'logistic' to"
                        " specify a linear or logistic trend.")
    parser.add_argument('--periods', dest='periods' , default=100, type=int,
                        help="int number of periods to forecast forward.")
    parser.add_argument('-plot_components', dest='plot_components' , action='store_true',
                        help="Plot the Prophet forecast components.")
    parser.add_argument('--coin', dest='coin', action='store', type=str,
                        help="The coin to forecast market symbol")
    args = parser.parse_args()
    print (type(args.periods))
    forecast(args)
    
if __name__== "__main__":
    main()