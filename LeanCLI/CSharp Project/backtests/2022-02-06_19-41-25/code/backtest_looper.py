import os



main_lines = ""

with open("tickers.txt","r") as ticker_file:
    ticker_data = ticker_file.readlines()
    
for i in range(len(ticker_data)):
    ticker = str(ticker_data[i])
        
    with open("OTHER_MAIN.txt", "r") as main_cs_file1:
        main_lines = main_cs_file1.readlines()
        main_lines[12] = '        private string _symbol = "' + ticker[:-1] + '";'
        print(main_lines[12])

    with open("main.cs", "w") as main_cs_file2:
        for i in range(len(main_lines)):
            main_cs_file2.write(main_lines[i])
            
    os.system("lean backtest main.cs")
ticker_file.close()
