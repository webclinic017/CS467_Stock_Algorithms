using CsvHelper;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;

namespace YahooToLean
{
    public class Converter
    {

        // Read the csv downloaded from yahoo finance
        public List<StockData> ReadData()
        {
            using (var reader = new StreamReader("D:\\StockTrading\\Data\\Yahoo\\EDC.csv"))    //C:\\Users\\chest\\Documents\\GitHub\\Lean\\Data\\equity\\usa\\daily\\tqqq\\TQQQ.csv"))
            using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
            {
                var records = new List<StockData>();
                csv.Read();
                csv.ReadHeader();
                while (csv.Read())
                {
                    var record = new StockData
                    {
                        Date = csv.GetField<DateTime>("Date"),
                        Open = csv.GetField<Double>("Open"),
                        High = csv.GetField<Double>("High"),
                        Low = csv.GetField<Double>("Low"),
                        Close = csv.GetField<Double>("Close"),
                        Volume = csv.GetField<int>("Volume")



                    };
                    records.Add(record);
                }
                return records;
            }
        }

        //Write the converted data to the lean format
        public void WriteData(List<StockData> data)
        {
            string fileName = @"D:\StockTrading\Data\Converted\EDC.csv";
            try
            {
                using (StreamWriter writer = new StreamWriter(fileName))
                {
                    foreach (var stockData in data)
                    {
                        string year = stockData.Date.Year.ToString();
                        string month = stockData.Date.Month.ToString("d2");
                        string day = stockData.Date.Day.ToString("d2");
                        string open = (Math.Round(stockData.Open,2) * 10000).ToString();
                        string high = (Math.Round(stockData.High,2) * 10000).ToString();
                        string low = (Math.Round(stockData.Low,2) * 10000).ToString();
                        string close = (Math.Round(stockData.Close,2) * 10000).ToString();
                        string volume = stockData.Volume.ToString();

                        writer.WriteLine(year + month + day + " 00:00"+"," + open + "," + high + "," + low + "," + close + "," + volume);
                    }
                }
            }
            catch (Exception exp)
            {
                Console.Write(exp.Message);
            }
            foreach (var stockData in data)
            {

            }
        }

        public void Convert()
        {
            List<StockData> data = ReadData();
            WriteData(data);
        }
    }
}
