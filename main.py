# Initial Network Testing

# Import Necessary Components

# For Data Parsing

import pandas as pd
import glob
import os
import numpy as np
from scipy.stats import linregress
import datetime
import matplotlib.pyplot as plt

# For Neural Networks

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns

# List CPU's Available
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
print(tf.__version__)

# Get CSV for Volatility and Momentum Calculations

ETFpath = '3x-ETF/'  # Should be updated to eventual folder name
ETFfile = 'TQQQ.csv'  # Should be updated to file name, or autosearch like "glob" functions below
ETF_3x = pd.read_csv(ETFpath + ETFfile)

# print("Here's the 3x ETF")
# print(ETF_3x)

# Calculating Historic Volatility from Raw 3x ETF Data

# Theory Yang-Zhang (OHLC)
# Sum of overnight volatility and weighted average of Rogers-Satchell volatility
# NOTE: In the Paper: MEASURING HISTORICAL VOLATILITY, temp_overnight_vol and temp_openclose_vol if
# implemented exactly would be 0 every time - it subtracts one value from its equal. Implementations of this
# aglorithm do not do that, so I did not either.


N = 2  # number of days in sample to consider leading up to current day, lowest it should be is 2

F = 1  # scaling factor

k = 0.34 / (1.34 + (N + 1) / (N - 1))

yang_zhang_vol_list = [np.nan] * N

for day in range(N, len(ETF_3x)):
    temp_rog_satch_vol = 0  # initialize to zero for every consideration
    temp_overnight_vol = 0  # initialize to zero for every consideration
    temp_openclose_vol = 0  # initialize to zero for every consideration
    for i in range(N):
        temp_rog_satch_vol += np.log(ETF_3x['High'][day - i] / ETF_3x['Close'][day - i]) * np.log(
            ETF_3x['High'][day - i] / ETF_3x['Open'][day - i]) + np.log(
            ETF_3x['Low'][day - i] / ETF_3x['Close'][day - i]) * np.log(
            ETF_3x['Low'][day - i] / ETF_3x['Open'][day - i])
        temp_overnight_vol += (np.log(ETF_3x['Open'][day - i] / ETF_3x['Close'][day - np.absolute((i - 1))])) ** 2
        temp_openclose_vol += (np.log(ETF_3x['Close'][day - i] / ETF_3x['Open'][day - i])) ** 2
    rog_satch_vol = temp_rog_satch_vol * F / N
    overnight_vol = temp_overnight_vol * F / (N - 1)
    openclose_vol = temp_openclose_vol * F / (N - 1)

    yang_zhang_vol = np.sqrt(F) * np.sqrt(overnight_vol + k * openclose_vol + (1 - k) * rog_satch_vol)

    yang_zhang_vol_list.append(yang_zhang_vol)

ETF_3x['Volatility'] = yang_zhang_vol_list

# print("Here's the 3x ETF with Volatility")
# print(ETF_3x)

# Calculating Momentum from Raw 3x ETF Data

# Used from Tutorial here: https://teddykoker.com/2019/05/momentum-strategy-from-stocks-on-the-move-in-python/

# Theory from Andreas Clenow

consideration_days = 30  # number of days you want take momentum over - would not recommend going less than 30, 90 is recommended by link and this number changes the output drastically
momentum_list = [np.nan] * consideration_days

for days in range(consideration_days, len(ETF_3x)):
    consideration_days_list = []
    for datapoint in range(consideration_days):
        consideration_days_list.append(ETF_3x['Close'][days - datapoint])
    returns = np.log(consideration_days_list)
    x = np.arange(len(consideration_days_list))
    slope, _, rvalue, _, _ = linregress(x, returns)

    momentum = (1 + slope) ** 252 * (rvalue ** 2)

    momentum_list.append(momentum)

ETF_3x['Momentum'] = momentum_list

# print("Here's the 3x ETF with Volatility and Momentum")
# print(ETF_3x)

# Sentimental Factors

# Calculating Put/Call Ratios

# help from here: https://www.geeksforgeeks.org/how-to-read-all-csv-files-in-a-folder-in-pandas/

PutCallpath = 'Put-Call-Ratio/'  # Should be updated to eventual folder name
PutCallfile = 'totalpc.csv'  # Should be updated to file name, or autosearch like "glob" functions below

# use if multiple CSV Files
# PutCallfiles = glob.glob(os.path.join(PutCallpath, "*.csv"))

PutCall_rawdata = pd.read_csv(PutCallpath + PutCallfile)

# print("Here's the Put/Call Ratio")
# print(PutCall_rawdata)

ETF_3x['Put/Call Ratio'] = np.nan

for days in range(len(ETF_3x)):
    etf_datetime = datetime.datetime.strptime(ETF_3x['Date'][days], "%Y-%m-%d")
    putcall_index = PutCall_rawdata[PutCall_rawdata['DATE'] == etf_datetime.strftime('%#m/%#d/%Y')].index.values
    ETF_3x['Put/Call Ratio'][days] = PutCall_rawdata['P/C Ratio'][putcall_index]

# print("Here's the 3x ETF with Volatility and Momentum and Put/Call Ratio")
# print(ETF_3x)

# Calculating Junk Bond Demand (Volume)

# Indicator for down days or up days not implemented

JunkBondpath = 'Junk-Bond-ETF/'  # Should be updated to eventual folder name
JunkBondfiles = glob.glob(os.path.join(JunkBondpath, "*.csv"))

frames = []
for junk_bonds_csvs in JunkBondfiles:
    temp_JunkBond = pd.read_csv(junk_bonds_csvs)
    frames.append(temp_JunkBond["Date"])
    frames.append(temp_JunkBond["Volume"])

headers = ["Date1", "Volume1", "Date2", "Volume2", "Date3", "Volume3", "Date4", "Volume4", "Date5",
           "Volume5"]  # hardcoded since requirement is "Top 5 ETF's to be Considered"
JunkBond_rawdata = pd.concat(frames, axis=1, keys=headers)
JunkBond_rawdata['Total Volume'] = np.nan

junkbond_datetime_start = datetime.date(2007, 1, 1)
junkbond_datetime_end = datetime.date.today()

days = junkbond_datetime_start
listpair = []

while days < junkbond_datetime_end:
    daily_volume = 0

    if len(JunkBond_rawdata[JunkBond_rawdata['Date1'] == days.strftime('%Y-%m-%d')].index.tolist()) > 0:
        index = JunkBond_rawdata[JunkBond_rawdata['Date1'] == days.strftime('%Y-%m-%d')].index.tolist()
        daily_volume += JunkBond_rawdata['Volume1'][index[0]]
    if len(JunkBond_rawdata[JunkBond_rawdata['Date2'] == days.strftime('%Y-%m-%d')].index.tolist()) > 0:
        index = JunkBond_rawdata[JunkBond_rawdata['Date2'] == days.strftime('%Y-%m-%d')].index.tolist()
        daily_volume += JunkBond_rawdata['Volume2'][index[0]]
    if len(JunkBond_rawdata[JunkBond_rawdata['Date3'] == days.strftime('%Y-%m-%d')].index.tolist()) > 0:
        index = JunkBond_rawdata[JunkBond_rawdata['Date3'] == days.strftime('%Y-%m-%d')].index.tolist()
        daily_volume += JunkBond_rawdata['Volume3'][index[0]]
    if len(JunkBond_rawdata[JunkBond_rawdata['Date4'] == days.strftime('%Y-%m-%d')].index.tolist()) > 0:
        index = JunkBond_rawdata[JunkBond_rawdata['Date4'] == days.strftime('%Y-%m-%d')].index.tolist()
        daily_volume += JunkBond_rawdata['Volume4'][index[0]]
    if len(JunkBond_rawdata[JunkBond_rawdata['Date5'] == days.strftime('%Y-%m-%d')].index.tolist()) > 0:
        index = JunkBond_rawdata[JunkBond_rawdata['Date5'] == days.strftime('%Y-%m-%d')].index.tolist()
        daily_volume += JunkBond_rawdata['Volume5'][index[0]]

    listpair.append([days.strftime('%Y-%m-%d'), daily_volume])

    days = days + datetime.timedelta(days=1)

JunkBond_interpreted_data = pd.DataFrame(listpair, columns=['Date', 'Volume'])

# print("Here's the Junk Bond Volume List, Interpreted")
# print(JunkBond_interpreted_data)

ETF_3x['Junk Bond Demand'] = np.nan

for days in range(len(ETF_3x)):
    etf_datetime = datetime.datetime.strptime(ETF_3x['Date'][days], "%Y-%m-%d")
    junkbond_index = JunkBond_interpreted_data[
        JunkBond_interpreted_data['Date'] == etf_datetime.strftime('%Y-%m-%d')].index.values
    ETF_3x['Junk Bond Demand'][days] = JunkBond_interpreted_data['Volume'][junkbond_index]

# print("Here's the 3x ETF with Volatility and Momentum and Put/Call Ratio and Junk Bond Demand")
# print(ETF_3x)

# Calculating McClellan Summation Index on S&P 500 and DOW

# Currently commented out because not enough data

# Sources:
# https://www.investopedia.com/terms/m/mcclellansummation.asp
# https://www.investopedia.com/terms/m/mcclellanoscillator.asp
# https://www.investopedia.com/terms/e/ema.asp


McClellanpath = 'McClellan-Summation-Index/' # Should be updated to eventual folder name
McClellanfiles = glob.glob(os.path.join(McClellanpath, "*.csv"))

frames = []
for index_files in McClellanfiles:
    temp_MIndex = pd.read_csv(index_files)
    frames.append(temp_MIndex["Date"])
    frames.append(temp_MIndex["Open"])
    frames.append(temp_MIndex["Close"])

headers = ["Date1", "Open1", "Close1", "Date2","Open2", "Close2"]
MIndex_rawdata = pd.concat(frames,axis=1,keys=headers)


# print("McClellan-Summation-Index Raw Data")
# print(MIndex_rawdata)

DOWadvance = []
SPXadvance = []
EMA19DOW = []
EMA19SPX = []
EMA39DOW = []
EMA39SPX = []

MCsummation_indexDOW = 0
MCsummation_indexSPX = 0

listpair_mc = []

# # Requires Date Old --> Date New
# # Adjusted for Size (normalized per Adjusted Oscillator formula)

for index in range(len(MIndex_rawdata)):

    # Calculate Advances and Declines for each day
    DOWadvance.append((MIndex_rawdata['Close1'][index] - MIndex_rawdata['Open1'][index]) /
                      (MIndex_rawdata['Close1'][index] + MIndex_rawdata['Open1'][index]))
    SPXadvance.append((MIndex_rawdata['Close2'][index] - MIndex_rawdata['Open2'][index]) /
                      (MIndex_rawdata['Close2'][index] + MIndex_rawdata['Open2'][index]))

    # Calculate simple and exponential moving averages (SMA and EMA)

    if index == 17:
        SMA19DOW = np.average(DOWadvance[0:index])
        SMA19SPX = np.average(SPXadvance[0:index])

    elif index == 18:
        EMA19DOW.append((DOWadvance[index]) - SMA19DOW * 0.1 + SMA19DOW)
        EMA19SPX.append((SPXadvance[index]) - SMA19SPX * 0.1 + SMA19SPX)

    elif index > 18:
        EMA19DOW.append((DOWadvance[index]) - EMA19DOW[index-19] * 0.1 + EMA19DOW[index-19])
        EMA19SPX.append((SPXadvance[index]) - EMA19SPX[index-19] * 0.1 + EMA19SPX[index-19])

        if index == 37:
            SMA39DOW = np.average(DOWadvance[0:index])
            SMA39SPX = np.average(SPXadvance[0:index])

        elif index == 38:
            EMA39DOW.append((DOWadvance[index]) - SMA39DOW * 0.05 + SMA19DOW)
            EMA39SPX.append((SPXadvance[index]) - SMA39SPX * 0.05 + SMA19SPX)

        elif index > 38:
            EMA39DOW.append((DOWadvance[index]) - EMA39DOW[index-39] * 0.05 + EMA39DOW[index-39])
            EMA39SPX.append((SPXadvance[index]) - EMA39SPX[index-39] * 0.05 + EMA39SPX[index-39])

            # Convert to McClellan Oscillator

            adjusted_MCOscillatorDOW = EMA19DOW[index-19] - EMA39DOW[index-39]
            adjusted_MCOscillatorSPX = EMA19SPX[index-19] - EMA39SPX[index-39]

            # Convert to McClellan Summation Index

            MCsummation_indexDOW = MCsummation_indexDOW + adjusted_MCOscillatorDOW
            MCsummation_indexSPX = MCsummation_indexSPX + adjusted_MCOscillatorSPX

            listpair_mc.append([MIndex_rawdata['Date1'][index], MCsummation_indexDOW, MIndex_rawdata['Date2'][index], MCsummation_indexSPX])


# Make McClellan Summation Index DataFrame
McCLellan_interpreted_data = pd.DataFrame(listpair_mc, columns = ['Date DOW', 'MC DOW', 'Date SPX', 'MC SPX'])

# print("McClellan-Summation-Index Interpreted Data")
# print(McCLellan_interpreted_data)

# Append to ETF Info

# ETF_3x['Dow McClellan Summation Index'] = np.nan
# ETF_3x['S&P McClellan Summation Index'] = np.nan

# for days in range(len(ETF_3x)):
#      etf_datetime = datetime.datetime.strptime(ETF_3x['Date'][days], "%Y-%m-%d")
#      mc_index = McCLellan_interpreted_data[McCLellan_interpreted_data['Date DOW'] ==
#      etf_datetime.strftime('%Y-%m-%d')].index.values
#      ETF_3x['Dow McClellan Summation Index'][days] = McCLellan_interpreted_data['MC DOW'][mc_index]

# for days in range(len(ETF_3x)):
#      etf_datetime = datetime.datetime.strptime(ETF_3x['Date'][days], "%Y-%m-%d")
#      mc_index = McCLellan_interpreted_data[McCLellan_interpreted_data['Date SPX'] ==
#      etf_datetime.strftime('%#m/%#d/%Y')].index.values
#      ETF_3x['S&P McClellan Summation Index'][days] = McCLellan_interpreted_data['MC SPX'][mc_index]

# Add in Quantity Traded Data

Quantpath = 'Algorithm_Backtests_All_3xETFs/datasets/' # Should be updated to eventual folder name
Quantfile = 'TQQQ-2010-03-01.csv'

# Three options to fit - Price, Quantity, Value - right now going to fit to just Quantity
# (since that's what we care about - Quantity we'll manipulate based on price)

# Note, for now, columns are off by one, so "Price" here equates to column

Quant_rawdata = pd.read_csv(Quantpath + Quantfile)
Quant_dates = Quant_rawdata.index

ETF_3x['Quantity Moved'] = np.nan

for days in range(len(ETF_3x)):
    flag = 0
    temp = 0
    for dates in Quant_dates:
        if ETF_3x['Date'][days] in dates:
            temp += float(Quant_rawdata['Price'][dates])

    if temp == 0:
        temp = np.nan
    ETF_3x['Quantity Moved'][days] = temp


# Add Volatility Time Lag Variables

num_days = 60  # starting value

delta_between = 2  # starting value

for n in range(num_days // delta_between):  # must be whole number
    ETF_3x['Volatility Time Lag ' + str(n)] = np.nan

    for days in range(len(ETF_3x)):
        if days >= n * delta_between:
            ETF_3x['Volatility Time Lag ' + str(n)][days] = ETF_3x['Volatility'][days - n * delta_between]

# Turn Date into Separate Column for Year, Month, and Date

ETF_3x['Year'] = np.nan
ETF_3x['Month'] = np.nan
ETF_3x['Day'] = np.nan

for days in range(len(ETF_3x)):
    etf_datetime = datetime.datetime.strptime(ETF_3x['Date'][days], "%Y-%m-%d")
    ETF_3x['Year'][days] = etf_datetime.strftime('%Y')
    ETF_3x['Month'][days] = etf_datetime.strftime('%m')
    ETF_3x['Day'][days] = etf_datetime.strftime('%d')

ETF_3x = ETF_3x.drop(columns=['Date'])


# Just removes all Nan values

ETF_3x = ETF_3x.dropna()
ETF_3x = ETF_3x.reset_index(drop=True)

# print("Here's the whole 3x ETF")
# print(ETF_3x)

# ETF_3x.to_csv('out.csv')

# Separating Training Data from Test Data

Train_X = ETF_3x.sample(frac=0.85, random_state=0)
Test_X = ETF_3x.drop(Train_X.index)

sns.pairplot(Train_X[['Volatility', 'Momentum', 'Put/Call Ratio', 'Junk Bond Demand']], diag_kind='kde')
print(Train_X.describe().transpose())

train_features = Train_X.copy()
test_features = Test_X.copy()

# mu, sigma = 3000, 67.0
# Train_Y = np.random.normal(mu, sigma, len(Train_X))
# Test_Y = np.random.normal(mu, sigma, len(Test_X))

train_labels = train_features.pop('Quantity Moved')
test_labels = test_features.pop('Quantity Moved')

print(Train_X.describe().transpose()[['mean', 'std']])

train_features=np.asarray(train_features).astype(float)
test_features=np.asarray(test_features).astype(float)

train_labels=np.asarray(train_labels).astype(float)
test_labels=np.asarray(test_labels).astype(float)

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())

test_model = keras.Sequential([
  normalizer,
  # layers.LSTM(4),
  layers.Dense(100, activation='relu'),
  layers.Dense(100, activation='relu'),
  layers.Dense(100, activation='relu'),
  layers.Dense(100, activation='relu'),
  layers.Dense(1)
])


test_model.compile(loss='mean_absolute_error',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
test_model.summary()

history = test_model.fit(
    train_features,
    train_labels,
    validation_split=0.15,
    verbose=0, epochs=1800)

print(history.history)

guess = test_model.predict(test_features)

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
# plt.plot(history.history['accuracy'], label='training accuracy')
# plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Error [Guess]')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
x = tf.linspace(0.0, len(test_labels)-1, len(test_labels))
plt.plot(x, test_labels, label='How Much Actually Traded')
plt.plot(x, guess, label='Guessed at How Much to Trade')
plt.legend()

plt.show()

test_results = {}
test_results['test_model'] = test_model.evaluate(test_features, test_labels, verbose=0)



# Next Step is to Change X vector time duration (default is 1 month) and Y value duration (default is 1 month)
# requires a nested For Loop to run these NN's








