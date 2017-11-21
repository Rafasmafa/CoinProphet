import argparse
import os
import pandas as pd
import numpy as np
from fbprophet import Prophet
from utils.exceptions import UnsupportedFileType

SUPPORTED_FILE_TYPES = ['.csv']


def determine_file_type(fn):
    ext = os.path.splitext(fn)[1]
    if ext in SUPPORTED_FILE_TYPES:
        return ext
    else:
        raise UnsupportedFileType('File type is not supported')

def forecast(args):
    file_type = determine_file_type(args.data)
    # TODO: make file extension parsing better
    # Not sure if other file types will be ok
    # This is just quick coding
    if file_type == '.csv':
        df = pd.read_csv(args.data)
    
    df['y'] = np.log(df['y'])
    df.head()
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=args.periods)
    future.tail()
    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    fig1 = m.plot(forecast)
    fig1.savefig('forecast.png')
    if args.plot_components:
        fig2 = m.plot_components(forecast)
        fig2.savefig('components.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',default=None, dest='data', required=True,
                        help='input data file. Supported file types: CSV')
    parser.add_argument('--growth', dest='growth', default=None,
                        help="String 'linear' or 'logistic' to"
                        " specify a linear or logistic trend.")
    parser.add_argument('--periods', dest='periods' , default=365, type=int,
                        help="int number of periods to forecast forward.")
    parser.add_argument('-plot_components', dest='plot_components' , action='store_true',
                        help="Plot the Prophet forecast components.")
    args = parser.parse_args()
    print (type(args.periods))
    forecast(args)
    
if __name__== "__main__":
    main()