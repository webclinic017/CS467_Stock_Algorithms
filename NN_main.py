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
import random

# For Neural Networks

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# List CPU's Available
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
print("Run with Tensorflow Version: " + tf.__version__)

# Get CSV for Volatility and Momentum Calculations

ETFpath = '3x-ETF/'  # Should be updated to eventual folder name
ETFfile = 'TQQQ.csv'  # Should be updated to file name, or autosearch like "glob" functions below
ETF_3x = pd.read_csv(ETFpath + ETFfile)

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

print("Volatility Calculated and Appended to Dataframe")
#
# Calculating Momentum from Raw 3x ETF Data

# Used from Tutorial here: https://teddykoker.com/2019/05/momentum-strategy-from-stocks-on-the-move-in-python/

# Theory from Andreas Clenow

consideration_days = 90  # number of days you want take momentum over - would not recommend going less than 30,
# 90 is recommended by link and this number changes the output drastically
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

print(str(consideration_days) + "-day Momentum Calculated and Appended to Dataframe")
#
# # Sentimental Factors
#
# # Calculating Put/Call Ratios
#
PutCallpath = 'Put-Call-Ratio/'  # Should be updated to eventual folder name
PutCallfile = 'totalpc.csv'  # Should be updated to file name, or autosearch like "glob" functions below

PutCall_rawdata = pd.read_csv(PutCallpath + PutCallfile)

ETF_3x['Put/Call Ratio'] = np.nan

for days in range(len(ETF_3x)):
    etf_datetime = datetime.datetime.strptime(ETF_3x['Date'][days], "%Y-%m-%d")
    putcall_index = PutCall_rawdata[PutCall_rawdata['DATE'] == etf_datetime.strftime('%#m/%#d/%Y')].index.values
    # ETF_3x['Put/Call Ratio'][days] = PutCall_rawdata['P/C Ratio'][putcall_index]
    if len(PutCall_rawdata.iloc[putcall_index, PutCall_rawdata.columns.get_loc('P/C Ratio')]) == 0:
        ETF_3x.iloc[days, ETF_3x.columns.get_loc('Put/Call Ratio')] = np.nan
    else:
        ETF_3x.iloc[days, ETF_3x.columns.get_loc('Put/Call Ratio')] = \
            PutCall_rawdata.iloc[putcall_index, PutCall_rawdata.columns.get_loc('P/C Ratio')]
#
print("Put/Call Ratio Compiled and Appended to Dataframe")

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

ETF_3x['Junk Bond Demand'] = np.nan

for days in range(len(ETF_3x)):
    etf_datetime = datetime.datetime.strptime(ETF_3x['Date'][days], "%Y-%m-%d")
    junkbond_index = JunkBond_interpreted_data[
        JunkBond_interpreted_data['Date'] == etf_datetime.strftime('%Y-%m-%d')].index.values
    #ETF_3x['Junk Bond Demand'][days] = JunkBond_interpreted_data['Volume'][junkbond_index]

    ETF_3x.iloc[days, ETF_3x.columns.get_loc('Junk Bond Demand')] = \
        JunkBond_interpreted_data.iloc[junkbond_index, JunkBond_interpreted_data.columns.get_loc('Volume')]


print("Junk Bond Demand Calculated and Appended to Dataframe")

# # Calculating McClellan Summation Index on stock in question
#
#
# # Sources:
# # https://www.investopedia.com/terms/m/mcclellansummation.asp
# # https://www.investopedia.com/terms/m/mcclellanoscillator.asp
# # https://www.investopedia.com/terms/e/ema.asp
# #
advance = []
EMA19 = []
EMA39 = []

MCsummation_index = 0
ETF_3x['McClellan Summation Index'] = np.nan

# Requires Date Old --> Date New
# Adjusted for Size (normalized per Adjusted Oscillator formula)

for index in range(len(ETF_3x)):

    # Calculate Advances and Declines for each day
    advance.append((ETF_3x['Close'][index] - ETF_3x['Open'][index]) /
                      (ETF_3x['Close'][index] + ETF_3x['Open'][index]))

    if index == 17:
        SMA19 = np.average(advance[0:index])

    elif index == 18:
        EMA19.append((advance[index]) - SMA19 * 0.1 + SMA19)

    elif index > 18:
        EMA19.append((advance[index]) - EMA19[index-19] * 0.1 + EMA19[index-19])

        if index == 37:
            SMA39 = np.average(advance[0:index])

        elif index == 38:
            EMA39.append((advance[index]) - SMA39 * 0.05 + SMA19)

        elif index > 38:
            EMA39.append((advance[index]) - EMA39[index-39] * 0.05 + EMA39[index-39])

            # Convert to McClellan Oscillator

            adjusted_MCOscillator = EMA19[index-19] - EMA39[index-39]

            # Convert to McClellan Summation Index

            MCsummation_index = MCsummation_index + adjusted_MCOscillator

            ETF_3x.iloc[index, ETF_3x.columns.get_loc('McClellan Summation Index')] = MCsummation_index

print("McClellan Summation Index Calculated and Appended to Dataframe")
#
# # Add in Profit for Traded Data
#
Quantpath = 'Algorithm_Backtests_All_3xETFs/datasets/' # Should be updated to eventual folder name
Quantfile = 'TQQQ-2010-03-01.csv'

# # Note, for now, columns are off by one, so "Price" here equates to column
#
Quant_rawdata = pd.read_csv(Quantpath + Quantfile)
Quant_dates = Quant_rawdata.index

ETF_3x['Profit Percentage'] = np.nan

start = 100000 # Assuming starting money amount is $100,000

for days in range(len(ETF_3x)):
    flag = 0
    temp = 0
    for dates in Quant_dates:
        if ETF_3x['Date'][days] in dates:
            temp += float(Quant_rawdata['Status'][dates])


    if temp == 0:
        temp_profit = np.nan
    else:
        temp_profit = temp / start * 100
        start += temp

    ETF_3x.iloc[days, ETF_3x.columns.get_loc('Profit Percentage')] = temp_profit

print("Profit Percentage by Trading Algorithm Compiled and Appended to Dataframe")
#
#
# Add Volatility Time Lag Variables

num_days = 5 # starting value

delta_between = 1  # starting value

for n in range(num_days // delta_between):  # must be whole number
    ETF_3x['Volatility Time Lag ' + str(n)] = np.nan

    for days in range(len(ETF_3x)):
        if days >= n * delta_between:
            # ETF_3x['Volatility Time Lag ' + str(n)][days] = ETF_3x['Volatility'][days - n * delta_between]
            ETF_3x.iloc[days, ETF_3x.columns.get_loc('Volatility Time Lag ' + str(n))] = \
                ETF_3x.iloc[days - n * delta_between, ETF_3x.columns.get_loc('Volatility')]


print(str(num_days) + "-day Lag with " + str(delta_between) + "-day Spacing Compiled and Appended to Dataframe")


# Turn Date into Separate Column for Year, Month, and Date

# ETF_3x['Year'] = np.nan
# ETF_3x['Month'] = np.nan
# ETF_3x['Day'] = np.nan
#
# for days in range(len(ETF_3x)):
#     etf_datetime = datetime.datetime.strptime(ETF_3x['Date'][days], "%Y-%m-%d")
#     ETF_3x.iloc[days, ETF_3x.columns.get_loc('Year')] = etf_datetime.strftime('%Y')
#     ETF_3x.iloc[days, ETF_3x.columns.get_loc('Month')] = etf_datetime.strftime('%m')
#     ETF_3x.iloc[days, ETF_3x.columns.get_loc('Day')] = etf_datetime.strftime('%d')

ETF_3x = ETF_3x.drop(columns=['Date', 'Open', 'Close', 'Adj Close'])

# Just removes all Nan values

ETF_3x = ETF_3x.dropna()
ETF_3x = ETF_3x.reset_index(drop=True)

print("The ETF Dataset has been Finalized and is Being Prepared for Neural Network Testing")

# Separating Training Data from Test Data

Train_X = ETF_3x.sample(frac=0.80, random_state=0)
Test_X = ETF_3x.drop(Train_X.index)

train_features = Train_X.copy()
test_features = Test_X.copy()

train_labels = train_features.pop('Profit Percentage')
test_labels = test_features.pop('Profit Percentage')

train_features=np.asarray(train_features).astype(float)
test_features=np.asarray(test_features).astype(float)

train_labels=np.asarray(train_labels).astype(float)
test_labels=np.asarray(test_labels).astype(float)

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

print("The Data has been prepared, the Neural Network is being Created")


test_model = keras.Sequential([
    normalizer,
    layers.Dense(10, activation='relu'),
    layers.Dense(10, activation='relu'),
    layers.Dense(1)
])

# https://keras.io/api/optimizers/learning_rate_schedules/exponential_decay/

initial_learning_rate = 0.0005
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.90,
    staircase=True)

test_model.compile(loss='mean_squared_error',
                   optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), metrics=['mse'])

print("The Neural Network has been Created and Compiled")

print("The Neural Network is Starting to Run Over the Training Data")

history = test_model.fit(
    train_features,
    train_labels,
    validation_split=0.20,
    verbose=0, epochs=200)


scores = test_model.evaluate(test_features, test_labels, verbose=0)

print("The Neural Network is Starting to Run over the Test Data")
guess = test_model.predict(test_features)

rand_guesses = []
for rand in range(len(test_labels)):
    rand_guesses.append(random.randint(-20,30))

print("Plotting Results")
plt.subplot(1,3,1)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Error [Guess]')
plt.legend()
plt.grid(True)

plt.subplot(1,3,2)
x = tf.linspace(0.0, len(test_labels)-1, len(test_labels))
plt.plot(x, test_labels, label='How Much Actually Traded')
plt.plot(x, guess, label='Network Guess at How Much Was Made')
plt.legend()

plt.subplot(1,3,3)
x = tf.linspace(0.0, len(test_labels)-1, len(test_labels))
plt.plot(x, test_labels, label='How Much Actually Traded')
plt.plot(x, rand_guesses, label='Random Guess at How Much Was Made')
plt.legend()

plt.show()

test_results = {}
test_results['test_model'] = test_model.evaluate(test_features, test_labels, verbose=0)








