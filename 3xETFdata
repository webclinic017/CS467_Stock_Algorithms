import yfinance as yf
import os

"""These are the 3x ETF tickers provided by: https://etfdb.com/themes/leveraged-3x-etfs/ """
tickers = ["TQQQ", "SOXL", "FAS", "UPRO", "SPXL", "TECL", "SQQQ", "TNA",
           "FNGU", "LABU", "UDOW", "ERX", "NUGT", "SPXU", "YINN", "DPST", "TZA", "SPXS", "TMF", "TMV", "URTY", "SDOW",
           "DFEN", "NAIL", "CURE", "TTT", "BRZU", "SRTY", "SOXS", "EDC", "DRN", "FAZ", "RETL", "FNGD", "TECS", "INDL",
           "MIDU", "TPOR", "YANG", "LABD", "UMDD", "SBND", "EURL", "DUSL", "KORU", "TYO", "EDZ", "DRV", "UBOT", "PILL",
           "ERY", "TYD", "UTSL", "OILU", "MEXX", "SMDD", "OILD"]

#These are 'junktickers' pulled from yahoo finance
junktickers = ["FTSL", "HYG", "JNK", "SRLN", "USHY"]


def get_3xETF_tickers():
    for name in tickers:
        ticker = yf.Ticker(name)

        # prints to console to show user progress
        print(ticker.info)

        # downloading historical market data
        hist = ticker.history(period="max")
        data_df = yf.download(name)

        # create 3X-ETF directory within the current directory if it doesn't already exist
        cur_dir = os.getcwd()
        out_dir = os.path.join(cur_dir, '3X-ETF')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        data_df.to_csv('3X-ETF/' + name + '.csv')

def get_junk_tickers():
    for name in junktickers:
        ticker = yf.Ticker(name)

        # prints to console to show user progress
        print(ticker.info)

        # get historical market data
        hist = ticker.history(period="max")
        data_df = yf.download(name)

        # create Junk-Bond-ETF directory within the current directory if it doesn't already exist
        cur_dir = os.getcwd()
        out_dir = os.path.join(cur_dir, 'Junk-Bond-ETF')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        data_df.to_csv('Junk-Bond-ETF/' + name + '.csv')


get_3xETF_tickers()
get_junk_tickers()
