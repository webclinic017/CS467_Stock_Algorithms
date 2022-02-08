from System.Drawing import Color
import weights as u


class FirstAlgo(QCAlgorithm):
    def Initialize(self):
        # Equity List
        self.equities = u.universe2
        self.weight = u.weight_1

        # Set starting information
        self.SetStartDate(2017, 1, 1)  # Set Start Date
        self.SetEndDate(2023, 1, 1)  # Set End Date
        self.SetCash(10000)  # Set Strategy Cash
        self.SetWarmUp(500)
        self._equity = None
        self._hours = 0

        # Technical Indicators
        self.moving_avg_close = {}
        self.moving_avg_high = {}
        self.moving_avg_low = {}
        self.moving_avg = {}
        self.bb = {}
        self.std = {}
        self.rocp = {}

        # Add Equities and Resolutions for Universe Selection
        for ticker in self.equities:
            self.AddEquity(ticker, Resolution.Hour)
        # self.AddEquity("SPY", Resolution.Hour)

        # Trade Information and variables
        self.weighted_resolutions = {}
        self.universal_max_weight = 100
        self.resolutions = self.weight.keys()
        self.trade = True

        # Scheduled Actions

        self.Schedule.On(self.DateRules.MonthStart(), self.TimeRules.At(1, 00), self.set_universe)
        # self.Schedule.On(self.DateRules.MonthStart(14), self.TimeRules.At(1, 00), self.set_universe)
        self.Schedule.On(self.DateRules.MonthStart(), self.TimeRules.At(1, 00), self.log_portfolio)

        # Build data dictionaries
        for ticker in self.equities:
            self.moving_avg_close[ticker] = {}
            self.moving_avg_high[ticker] = {}
            self.moving_avg_low[ticker] = {}
            self.moving_avg[ticker] = {}
            self.bb[ticker] = {}
            self.std[ticker] = {}
            self.rocp[ticker] = {}
            for resolution in self.resolutions:
                self.moving_avg_close[ticker][resolution] = self.SMA(ticker, resolution, Resolution.Daily, Field.Close)
                self.moving_avg_high[ticker][resolution] = self.SMA(ticker, resolution, Resolution.Daily, Field.High)
                self.moving_avg_low[ticker][resolution] = self.SMA(ticker, resolution, Resolution.Daily, Field.Low)
                self.moving_avg[ticker][resolution] = self.SMA(ticker, resolution, Resolution.Daily)
            self.bb[ticker] = self.BB(ticker, 14, Resolution.Daily)
            self.std[ticker] = self.STD(ticker, 10, Resolution.Daily)
            self.rocp[ticker] = self.ROCP(ticker, 62, Resolution.Daily)

        # Set Initial Universe
        if self._equity is None:
            self.set_universe()

    def OnData(self, data):
        """
            OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
            Arguments:
                data: Slice object keyed by symbol containing the stock data
        """
        # Validate Data and set Daily Data

        if self.IsWarmingUp is True or self._equity is None:
            return

        re_balance = False

        if not data.Bars.ContainsKey(self._equity):
            return

        bar = data.Bars[self._equity]

        # Buy and Sell Signals
        for resolution in self.resolutions:

            # Buy Signals
            if bar.Close > self.moving_avg[self._equity][resolution].Current.Value and \
                    self.weighted_resolutions[self._equity][resolution]["weight"] != \
                    self.weighted_resolutions[self._equity][resolution]["max_weight"] and \
                    self.weighted_resolutions[self._equity][resolution]["max_weight"] != 0 and self.trade is True:

                # Increase weight for momentum buying
                self.weighted_resolutions[self._equity][resolution]["weight"] = \
                    self.weighted_resolutions[self._equity][resolution]["max_weight"]
                re_balance = True

            # Sell Signal
            elif bar.Close < self.moving_avg_low[self._equity][resolution].Current.Value and \
                    self.weighted_resolutions[self._equity][resolution]["weight"] != 0 and \
                    self.weighted_resolutions[self._equity][resolution][
                        "max_weight"] != 0 and self.trade is True:  # SELF TRADE

                # Decrease weight for momentum selling
                self.weighted_resolutions[self._equity][resolution]["weight"] = 0
                re_balance = True
                # self.Log(bar)

        # Make trades if momentum dictates
        if re_balance is True:
            self.SetHoldings(self._equity, self.buy_signals(self._equity))

    def buy_signals(self, ticker):
        """
        Calculates buy signals for ticker
        """
        buy = 0

        for resolution in self.resolutions:
            buy += self.weighted_resolutions[ticker][resolution]["weight"]

        return buy / self.universal_max_weight

    def set_universe(self):
        """
        This method chooses from the available tickers and sets the universe
        """
        active = []
        top = None

        # Get the tickers that have a positive rate of change
        for ticker in self.equities:
            if self.std[ticker].Current.Value != 0:
                active.append(ticker)

        if len(active) > 0:
            top = active[0]

        for ticker in active:
            # self.Log("---{0} STD {1}--- STD PERCENT {2}".format(ticker, self.std[ticker].Current.Value, self.std[ticker].Current.Value/self.moving_avg[ticker][3].Current.Value))
            if self.std[ticker].Current.Value / self.moving_avg[ticker][10].Current.Value > self.std[
                top].Current.Value / self.moving_avg[top][10].Current.Value:
                top = ticker

        self._equity = top
        self.weighted_resolutions[top] = self.weight

        # If ticker is not in focus liquidate it
        for ticker in self.equities:
            if ticker is not self._equity:
                self.Liquidate(ticker)

    def log_portfolio(self):
        """
        Writes portfolio to log
        """
        invested_tickers = 0
        self.Log("PORTFOLIO TOTAL: {0} || TOTAL MARGIN USED: {1} || IS TRADING: {2}".format(
            self.Portfolio.TotalPortfolioValue, self.Portfolio.TotalMarginUsed, self.trade))
        for ticker in self.equities:
            if self.ActiveSecurities[ticker].Invested is True:
                invested_tickers += 1

        self.Log("INVESTED TICKERS: {0}".format(invested_tickers))
        self.Log("EQUITIES: {0}".format(self._equity))