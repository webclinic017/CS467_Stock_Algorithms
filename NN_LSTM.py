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
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# List CPU's Available
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
print("Run with Tensorflow Version: " + tf.__version__)

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


N = 5  # number of days in sample to consider leading up to current day, lowest it should be is 2

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

# print("Here's the 3x ETF with Volatility")
# print(ETF_3x)

# Calculating Momentum from Raw 3x ETF Data

# Used from Tutorial here: https://teddykoker.com/2019/05/momentum-strategy-from-stocks-on-the-move-in-python/

# Theory from Andreas Clenow

consideration_days = 90  # number of days you want take momentum over - would not recommend going less than 30, 90 is recommended by link and this number changes the output drastically
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
    # ETF_3x['Put/Call Ratio'][days] = PutCall_rawdata['P/C Ratio'][putcall_index]
    if len(PutCall_rawdata.iloc[putcall_index, PutCall_rawdata.columns.get_loc('P/C Ratio')]) == 0:
        ETF_3x.iloc[days, ETF_3x.columns.get_loc('Put/Call Ratio')] = np.nan
    else:
        ETF_3x.iloc[days, ETF_3x.columns.get_loc('Put/Call Ratio')] = \
            PutCall_rawdata.iloc[putcall_index, PutCall_rawdata.columns.get_loc('P/C Ratio')]

print("Put/Call Ratio Compiled and Appended to Dataframe")

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

    # JunkBond_rawdata.iloc[JunkBond_rawdata['Date1']]

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
    #ETF_3x['Junk Bond Demand'][days] = JunkBond_interpreted_data['Volume'][junkbond_index]

    ETF_3x.iloc[days, ETF_3x.columns.get_loc('Junk Bond Demand')] = \
        JunkBond_interpreted_data.iloc[junkbond_index, JunkBond_interpreted_data.columns.get_loc('Volume')]


print("Junk Bond Demand Calculated and Appended to Dataframe")

# print("Here's the 3x ETF with Volatility and Momentum and Put/Call Ratio and Junk Bond Demand")
# print(ETF_3x)

# Calculating McClellan Summation Index on stock in question


# Sources:
# https://www.investopedia.com/terms/m/mcclellansummation.asp
# https://www.investopedia.com/terms/m/mcclellanoscillator.asp
# https://www.investopedia.com/terms/e/ema.asp


# McClellanpath = 'McClellan-Summation-Index/' # Should be updated to eventual folder name
# McClellanfiles = glob.glob(os.path.join(McClellanpath, "*.csv"))
#
# frames = []
# for index_files in McClellanfiles:
#     temp_MIndex = pd.read_csv(index_files)
#     frames.append(temp_MIndex["Date"])
#     frames.append(temp_MIndex["Open"])
#     frames.append(temp_MIndex["Close"])
#
# headers = ["Date1", "Open1", "Close1", "Date2","Open2", "Close2"]
# MIndex_rawdata = pd.concat(frames,axis=1,keys=headers)
#
#
# # print("McClellan-Summation-Index Raw Data")
# # print(MIndex_rawdata)
#
advance = []
# SPXadvance = []
EMA19 = []
# EMA19SPX = []
EMA39 = []
# EMA39SPX = []

MCsummation_index = 0
ETF_3x['McClellan Summation Index'] = np.nan

listpair_mc = []

# Requires Date Old --> Date New
# Adjusted for Size (normalized per Adjusted Oscillator formula)

for index in range(len(ETF_3x)):

    # Calculate Advances and Declines for each day
    advance.append((ETF_3x['Close'][index] - ETF_3x['Open'][index]) /
                      (ETF_3x['Close'][index] + ETF_3x['Open'][index]))
    # SPXadvance.append((MIndex_rawdata['Close2'][index] - MIndex_rawdata['Open2'][index]) /
    #                   (MIndex_rawdata['Close2'][index] + MIndex_rawdata['Open2'][index]))

    # Calculate simple and exponential moving averages (SMA and EMA)

    if index == 17:
        SMA19 = np.average(advance[0:index])
        # SMA19SPX = np.average(SPXadvance[0:index])

    elif index == 18:
        EMA19.append((advance[index]) - SMA19 * 0.1 + SMA19)
        # EMA19SPX.append((SPXadvance[index]) - SMA19SPX * 0.1 + SMA19SPX)

    elif index > 18:
        EMA19.append((advance[index]) - EMA19[index-19] * 0.1 + EMA19[index-19])
        # EMA19SPX.append((SPXadvance[index]) - EMA19SPX[index-19] * 0.1 + EMA19SPX[index-19])

        if index == 37:
            SMA39 = np.average(advance[0:index])
            # SMA39SPX = np.average(SPXadvance[0:index])

        elif index == 38:
            EMA39.append((advance[index]) - SMA39 * 0.05 + SMA19)
            # EMA39SPX.append((SPXadvance[index]) - SMA39SPX * 0.05 + SMA19SPX)

        elif index > 38:
            EMA39.append((advance[index]) - EMA39[index-39] * 0.05 + EMA39[index-39])
            # EMA39SPX.append((SPXadvance[index]) - EMA39SPX[index-39] * 0.05 + EMA39SPX[index-39])

            # Convert to McClellan Oscillator

            adjusted_MCOscillator = EMA19[index-19] - EMA39[index-39]
            # adjusted_MCOscillatorSPX = EMA19SPX[index-19] - EMA39SPX[index-39]

            # Convert to McClellan Summation Index

            MCsummation_index = MCsummation_index + adjusted_MCOscillator
            # MCsummation_indexSPX = MCsummation_indexSPX + adjusted_MCOscillatorSPX

            #listpair_mc.append([MIndex_rawdata['Date1'][index], MCsummation_indexDOW, MIndex_rawdata['Date2'][index], MCsummation_indexSPX])
            # listpair_mc.append([ETF_3x['Date'][index], MCsummation_indexDOW, MIndex_rawdata['Date2'][index],
            #                     MCsummation_indexSPX])
            ETF_3x.iloc[index, ETF_3x.columns.get_loc('McClellan Summation Index')] = MCsummation_index

print("McClellan Summation Index Calculated and Appended to Dataframe")

# Make McClellan Summation Index DataFrame
# McCLellan_interpreted_data = pd.DataFrame(listpair_mc, columns = ['Date DOW', 'MC DOW', 'Date SPX', 'MC SPX'])

# print("McClellan-Summation-Index Interpreted Data")
# print(McCLellan_interpreted_data)

# Append to ETF Info

# ETF_3x['Dow McClellan Summation Index'] = np.nan
# ETF_3x['S&P McClellan Summation Index'] = np.nan
#
# for days in range(len(ETF_3x)):
#      etf_datetime = datetime.datetime.strptime(ETF_3x['Date'][days], "%Y-%m-%d")
#      mc_index = McCLellan_interpreted_data[McCLellan_interpreted_data['Date DOW'] ==
#      etf_datetime.strftime('%Y-%m-%d')].index.values
#
#      ETF_3x.iloc[days, ETF_3x.columns.get_loc('Dow McClellan Summation Index')] = \
#         McCLellan_interpreted_data.iloc[mc_index, McCLellan_interpreted_data.columns.get_loc('MC DOW')]
#
#      # ETF_3x['Dow McClellan Summation Index'][days] = McCLellan_interpreted_data['MC DOW'][mc_index]
#
# for days in range(len(ETF_3x)):
#      etf_datetime = datetime.datetime.strptime(ETF_3x['Date'][days], "%Y-%m-%d")
#      mc_index = McCLellan_interpreted_data[McCLellan_interpreted_data['Date SPX'] ==
#      etf_datetime.strftime('%#m/%#d/%Y')].index.values
#      ETF_3x.iloc[days, ETF_3x.columns.get_loc('S&P McClellan Summation Index')] = \
#                  McCLellan_interpreted_data.iloc[mc_index, McCLellan_interpreted_data.columns.get_loc('MC SPX')]
#      # ETF_3x['S&P McClellan Summation Index'][days] = McCLellan_interpreted_data['MC SPX'][mc_index]

# Add in Quantity Traded Data

Quantpath = 'Algorithm_Backtests_All_3xETFs/datasets/' # Should be updated to eventual folder name
Quantfile = 'TQQQ-2010-03-01.csv'

# Three options to fit - Price, Quantity, Value - right now going to fit to just Quantity
# (since that's what we care about - Quantity we'll manipulate based on price)

# Note, for now, columns are off by one, so "Price" here equates to column

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
        temp = np.nan
    else:
        temp_profit = temp / start
        start += temp

    #ETF_3x['Quantity Moved'][days] = temp
    ETF_3x.iloc[days, ETF_3x.columns.get_loc('Profit Percentage')] = temp

print("Profit Percentage by Trading Algorithm Compiled and Appended to Dataframe")


# Add Volatility Time Lag Variables
#
# num_days = 30  # starting value
#
# delta_between = 6  # starting value
#
# for n in range(num_days // delta_between):  # must be whole number
#     ETF_3x['Volatility Time Lag ' + str(n)] = np.nan
#
#     for days in range(len(ETF_3x)):
#         if days >= n * delta_between:
#             # ETF_3x['Volatility Time Lag ' + str(n)][days] = ETF_3x['Volatility'][days - n * delta_between]
#             ETF_3x.iloc[days, ETF_3x.columns.get_loc('Volatility Time Lag ' + str(n))] = \
#                 ETF_3x.iloc[days - n * delta_between, ETF_3x.columns.get_loc('Volatility')]
#
#
# print(str(num_days) + "-day Lag with " + str(delta_between) + "-day Spacing Compiled and Appended to Dataframe")


# Turn Date into Separate Column for Year, Month, and Date
#
# ETF_3x['Year'] = np.nan
# ETF_3x['Month'] = np.nan
# ETF_3x['Day'] = np.nan
#
# for days in range(len(ETF_3x)):
#     etf_datetime = datetime.datetime.strptime(ETF_3x['Date'][days], "%Y-%m-%d")
#     ETF_3x.iloc[days, ETF_3x.columns.get_loc('Year')] = etf_datetime.strftime('%Y')
#     ETF_3x.iloc[days, ETF_3x.columns.get_loc('Month')] = etf_datetime.strftime('%m')
#     ETF_3x.iloc[days, ETF_3x.columns.get_loc('Day')] = etf_datetime.strftime('%d')
#     # ETF_3x['Year'][days] = etf_datetime.strftime('%Y')
#     # ETF_3x['Month'][days] = etf_datetime.strftime('%m')
#     # ETF_3x['Day'][days] = etf_datetime.strftime('%d')

# ETF_3x = ETF_3x.drop(columns=['Date', 'Open', 'Low', 'Adj Close', 'Junk Bond Demand', 'McClellan Summation Index'])
# ETF_3x = ETF_3x.drop(columns=['Date'])

# Just removes all Nan values

ETF_3x = ETF_3x.dropna()
ETF_3x = ETF_3x.reset_index(drop=True)

print("The ETF Dataset has been Finalized and is Being Prepared for Neural Network Testing")

# print("Here's the whole 3x ETF")
# print(ETF_3x)
#
# ETF_3x.to_csv('out_with_NAs.csv')

# Separating Training Data from Test Data

Train_X = ETF_3x.sample(frac=0.80, random_state=0)
Test_X = ETF_3x.drop(Train_X.index)

# sns.pairplot(Train_X[['Volatility', 'Momentum', 'Put/Call Ratio', 'Junk Bond Demand']], diag_kind='kde')
# print(Train_X.describe().transpose())

train_features = Train_X.copy()
test_features = Test_X.copy()

# mu, sigma = 3000, 67.0
# Train_Y = np.random.normal(mu, sigma, len(Train_X))
# Test_Y = np.random.normal(mu, sigma, len(Test_X))

train_labels = train_features.pop('Profit Percentage')
test_labels = test_features.pop('Profit Percentage')

sc = MinMaxScaler(feature_range = (0, 1))
scaled_train = sc.fit_transform(train_features)

X_train = []
y_train = []
for i in range(60, len(X_train)):
    X_train.append(scaled_train[i-60:i, 0])
    y_train.append(scaled_train[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# print(Train_X.describe().transpose()[['mean', 'std']])

# train_features=np.asarray(train_features).astype(float)
# test_features=np.asarray(test_features).astype(float)
#
# train_labels=np.asarray(train_labels).astype(float)
# test_labels=np.asarray(test_labels).astype(float)
#
# normalizer = tf.keras.layers.Normalization(axis=-1)
# normalizer.adapt(np.array(train_features))
# print(normalizer.mean.numpy())

first = np.array(train_features[:1])

# with np.printoptions(precision=2, suppress=True):
#   print('First example:', first)
#   print()
#   print('Normalized:', normalizer(first).numpy())

# train_features = np.reshape(train_features, (train_features.shape[0], train_features.shape[1]), 1)

print("The Data has been prepared, the Neural Network is being Created")



print("The Neural Network has been Created and Compiled")

# print("About the model:")
# test_model.summary()

print("The Neural Network is Starting to Run Over the Training Data")

# K-Fold Validation: https://www.machinecurve.com/index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/

num_folds = 5
loss_per_fold = []
acc_per_fold = []

kfold_input = np.concatenate((train_features, test_features), axis = 0)
kfold_target = np.concatenate((train_labels, test_labels), axis = 0)

kfold = KFold(n_splits=num_folds, shuffle=True)

fold_no = 1

# for train, test in kfold.split(kfold_input, kfold_target):

# print("K-Fold Run Number: " + str(fold_no))
test_model = keras.Sequential([
    # normalizer,
    layers.LSTM(units=50, return_state=True, input_shape=(X_train.shape[1], 1)),
    layers.Dropout(0.2),
    layers.LSTM(units=50, return_state=True, input_shape=(X_train.shape[1], 1)),
    layers.Dropout(0.2),
    layers.LSTM(units=50, return_state=True, input_shape=(X_train.shape[1], 1)),
    layers.Dropout(0.2),
    # layers.Dense(40, activation='relu'),
    layers.Dense(1)
])

# https://keras.io/api/optimizers/learning_rate_schedules/exponential_decay/

initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True)

test_model.compile(loss='mean_absolute_error',
                   optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), metrics=['mae'])

history = test_model.fit(
    train_features,
    train_labels,
    validation_split=0.20,
    verbose=0, epochs=180)


scores = test_model.evaluate(test_features, test_labels, verbose=0)
# print(f'Test loss: {scores}')
loss_per_fold.append(scores)

fold_no = fold_no + 1

# print(history.history)

# == Provide average scores ==
# print('------------------------------------------------------------------------')
# print('Score per fold')
# for i in range(0, len(loss_per_fold)):
#   print('------------------------------------------------------------------------')
#   print(f'> Fold {i+1} - Loss: {loss_per_fold[i]}')
# print('------------------------------------------------------------------------')
# print('Average scores for all folds:')
# print(f'> Loss: {np.mean(loss_per_fold)}')
# print('------------------------------------------------------------------------')

print("The Neural Network is Starting to Run over the Test Data")
guess = test_model.predict(test_features)


print("Plotting Results")
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








