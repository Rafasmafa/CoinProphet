import pandas as pd
import numpy as np
from fbprophet import Prophet


df = pd.read_csv('example_data/peyton.csv')
df['y'] = np.log(df['y'])
df.head()
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=365)
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)
fig1.savefig('plot1.png')
fig2 = m.plot_components(forecast)
fig2.savefig('plot2.png')
