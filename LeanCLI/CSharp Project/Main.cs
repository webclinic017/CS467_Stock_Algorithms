using QuantConnect.Data;
using QuantConnect.Data.Market;
using QuantConnect.Indicators;
using System;
using System.Collections.Generic;
using System.Linq;

namespace QuantConnect.Algorithm.CSharp
{
    public class MultiResolutionAlgorithm : QCAlgorithm
    {

        private string _symbol = "GIS";        //private DateTime _previous;

        public List<SimpleMovingAverage> movingAveragesClose;
        public List<SimpleMovingAverage> movingAveragesHigh;
        public List<SimpleMovingAverage> movingAveragesLow;


        public List<int> maLengths;
        public int[] tradeStatus;

        public override void Initialize()
        {
            SetStartDate(1975, 1, 1);  //Set Start Date
            SetEndDate(2023, 12, 30);    //Set End Date
            SetCash(100000);             //Set Strategy 
            SetWarmup(100);

            movingAveragesClose = new List<SimpleMovingAverage>();
            movingAveragesHigh = new List<SimpleMovingAverage>();
            movingAveragesLow = new List<SimpleMovingAverage>();


            AddEquity(_symbol, Resolution.Daily);

            maLengths = new List<int>() { 5, 10, 20, 62 };
            tradeStatus = new int[maLengths.Count];
            for (int i = 0; i < maLengths.Count; i++)
            {
                movingAveragesClose.Add(SMA(_symbol, maLengths[i], Resolution.Daily, x => ((TradeBar)x).Close));
                movingAveragesHigh.Add(SMA(_symbol, maLengths[i], Resolution.Daily, x => ((TradeBar)x).High));
                movingAveragesLow.Add(SMA(_symbol, maLengths[i], Resolution.Daily, x => ((TradeBar)x).Low));

            }
        }

        /// <summary>
        /// OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
        /// </summary>
        /// <param name="data">Slice object keyed by symbol containing the stock data</param>
        public override void OnData(Slice data)
        {
            if (IsWarmingUp) return;

            var holdings = Portfolio[_symbol].Quantity;

            TradeBar bar = data.Bars[_symbol];

            // Figure out the current signal for the different resolutions
            bool changed = false;
            for (int i = 0; i < maLengths.Count; i++)
            {
                //new buy signal
                if (bar.Close > movingAveragesHigh[i] && tradeStatus[i] == 0)
                {
                    tradeStatus[i] = 1;
                    changed = true;
                }
                //new sell signal
                else if (bar.Close < movingAveragesLow[i] && tradeStatus[i] == 1)
                {
                    tradeStatus[i] = 0;
                    changed = true;
                }

            }

            // Set the trades
            if (changed)
            {
                // get the number of buy signals
                double buySignals = tradeStatus.Sum();
                var percentage = buySignals / maLengths.Count;
                SetHoldings(_symbol, percentage);
            }

        }

    }
}