import requests
import alpha_vantage
import numpy as np
import pandas as pd
from numpy.random import randn
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import pandas_datareader.data as web
import pandas_datareader
import datetime
import math
from pandas_datareader.yahoo.options import Options as YahooOptions
from pandas.plotting import scatter_matrix
from matplotlib.dates import DateFormatter, date2num, WeekdayLocator, DayLocator, MONDAY
from mpl_finance import candlestick_ohlc
from pandas_datareader import data
from yahoofinancials import YahooFinancials
from dateutil.relativedelta import relativedelta
import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
from dateutil import relativedelta
import time
import talib
from lxml import html
import requests
import json
import argparse
from bs4 import BeautifulSoup
from collections import OrderedDict
import warnings


class realDCF:    #performs the DCF calculation per share
    def dcf(self):
        warnings.filterwarnings("ignore")
        stocks = ["AAPL"]
        for ticker in stocks:
            print("Fetching data for", ticker, "....")
            print(getBeta(ticker))
            exchangeName = getExchange(ticker)
            dict = getKeyData(ticker)
            debt = float(getDebt(ticker))
            forecast = getForecasts(ticker)
            WACC = getWACC(ticker, debt, dict)
            print("The forecast of future free cash flows for the next 5y is", forecast)
            print("The WACC used for this model was: ", WACC)
            forecast.remove(forecast[0])
            print("The estimates growth rate for the next 5y is:", forecast[1]/forecast[0])
            forecastTime = []
            for i in range(0, len(forecast)):
                discounted = forecast[i] / (WACC) ** (i + 1)
                forecastTime.append(discounted)
            lastForecast = forecast[-1]
            growth = 1.02
            terminal_value = lastForecast / (WACC - growth)
            if math.isnan(terminal_value):
                terminal_value = 0
            PV_terminal_value = int(terminal_value / (WACC) ** 6)
            if PV_terminal_value < 0 and forecastTime[-1] > 0:
                PV_terminal_value = 0
            enterprise_value = sum(forecastTime) + PV_terminal_value
            equity_value = enterprise_value - debt
            shares = dict['shares'] * 1000000
            fair_value = equity_value / (shares)
            print("The fair value for", ticker, "is", fair_value, exchangeName)
            print("The market price for", ticker, "is", dict['mp'], exchangeName)
            if dict['mp'] > fair_value:
                premium = ((dict['mp'] / fair_value)-1) * 100
                print(ticker, "trades at a premium of", abs(premium), "percent to fair value")
            else:
                discount = ((fair_value / dict['mp']) - 1) * 100.0
                print(ticker, "trades at a discount of", abs(discount), "percent to fair value")


#Returns dict containing market cap, total debt, beta, total equity, eps, interest expense, current market price,
#shares outstanding, etc.
def getKeyData(ticker):
    url = "https://stockanalysis.com/stocks/{}/".format(ticker)
    response = requests.get(url, verify=False)
    parser = html.fromstring(response.content)
    shares = parser.xpath('//div[@class="info"]//table//tbody//tr[td/text()[contains(., "Shares Out")]]')

    shares = shares[0].xpath('td/text()')[1]
    factor = 1000 if 'B' in shares else 1
    shares = float(shares.replace('B', '').replace('M', '')) * factor

    url = "https://stockanalysis.com/stocks/{}/financials/".format(ticker)
    response = requests.get(url, verify=False)
    parser = html.fromstring(response.content)
    market_price = float(parser.xpath('//div[@id="sp"]/span[@id="cpr"]/text()')[0].replace('$', '').replace(',', ''))
    return {'mp': market_price, 'shares': shares}

def getBeta(ticker):
    url = "https://finance.yahoo.com/quote/{}/".format(ticker)
    statistics = pd.read_html(url)
    beta = float(statistics[1][1][1])
    if (math.isnan(beta)):
        beta = 1
    return beta


#Returns dataframe representing historical free cash flows
def getForecasts(ticker):
    API_URL = "https://www.alphavantage.co/query?"
    data = {
        "function": "CASH_FLOW",
        "symbol": ticker,
        "outputsize": "compact",
        "apikey": "ANR8W8CH92FTSZLQ"
    }
    response = requests.get(API_URL, params=data)
    html_file = BeautifulSoup(response.text, "html.parser")
    url = html_file.prettify()
    url = json.loads(url)
    exchange = (url['annualReports'][0])
    factor = getExchangeFactor(exchange['reportedCurrency'])
    cashFlows = []
    for x in range (1):
        firstReport = (url['annualReports'][x])
        opCF = float(firstReport['operatingCashflow']) * factor
        capex = firstReport['capitalExpenditures']
        if capex == "None":
            capex = 0.0
        capex = float(capex) * factor
        cashFlows.append(opCF - capex)
    latest_free_cash_flow_firm = float(cashFlows[0])
    earliest_free_cash_flow_firm = float(cashFlows[-1])
    CAGR = (latest_free_cash_flow_firm / earliest_free_cash_flow_firm * 100) / 5
    otherCAGR = getFutureGrowth(ticker)
    if otherCAGR != 0:
        CAGR = otherCAGR
    CAGR = CAGR/100 + 1
    forecast = [1,2,3,4,5,6]
    forecast[0] = latest_free_cash_flow_firm #store latest cash flow of the firm to start projection
    if latest_free_cash_flow_firm < 0 and CAGR > 0:
        CAGR = CAGR - 1
    for i in range (1,6):
        if forecast[i-1] < 0 and CAGR < 0: #handle negative CAGR and negative FCF
            CAGR *= -1
        forecast[i] = forecast[i-1] * CAGR
        if (CAGR >= 3):
            CAGR /= 2
        if (i >= 3 and CAGR >= 1.5):
            CAGR = CAGR * 0.98
    return forecast


#Returns analysts' 5y future growth expectations in earnings
def getFutureGrowth(ticker):
    url = "https://in.finance.yahoo.com/quote/{}/analysis?p={}".format(ticker, ticker)
    response = requests.get(url, verify=False)
    parser = html.fromstring(response.content)
    ge = parser.xpath('//table//tbody//tr')
    for row in ge:
        label = row.xpath("td/span/text()")[0]
        if 'Next 5 years' in label:
            if (row.xpath("td/text()")[0] != 'N/A'):
                ge = float(row.xpath("td/text()")[0].replace('%', ''))

    if (isinstance(ge, float)):
        return ge
    else:
        return 0

#Calculates cost of equity
def costOfEquity(ticker):
    beta = getBeta(ticker)
    riskFree = 1.51
    marketPremium = 6.0
    costofEquity = riskFree + (beta * (marketPremium - riskFree))
    return costofEquity


#Calculates cost of debt
def costOfDebt(ticker, debt):
    API_URL = "https://www.alphavantage.co/query?"
    data = {
        "function": "INCOME_STATEMENT",
        "symbol": ticker,
        "outputsize": "compact",
        "apikey": "ANR8W8CH92FTSZLQ"
    }
    response = requests.get(API_URL, params=data)
    html_file = BeautifulSoup(response.text, "html.parser")
    url = html_file.prettify()
    url = json.loads(url)
    firstReport = (url['annualReports'][0])
    interestExpense = (firstReport['interestExpense'])
    if (interestExpense == "None"):
        interestExpense = 0.0
    exchange = getExchange(ticker)
    interestExpense = float(interestExpense) * getExchangeFactor(exchange)
    numerator = interestExpense * (1-.2) * 100
    debt = float(debt)
    if (debt == 0):
        return 1
    return numerator/debt

def getExchange(ticker):
    API_URL = "https://www.alphavantage.co/query?"
    data = {
        "function": "CASH_FLOW",
        "symbol": ticker,
        "outputsize": "compact",
        "apikey": "ANR8W8CH92FTSZLQ"
    }
    response = requests.get(API_URL, params=data)
    html_file = BeautifulSoup(response.text, "html.parser")
    url = html_file.prettify()
    url = json.loads(url)
    exchange = (url['annualReports'][0])
    exchange = exchange['reportedCurrency']
    return exchange

def getDebt(ticker):
    exchange = getExchange(ticker)
    factor = getExchangeFactor(exchange)
    url = "https://finance.yahoo.com/quote/{}/balance-sheet/".format(ticker)
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cache-Control': 'max-age=0',
        'Pragma': 'no-cache',
        'Referrer': 'https://google.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36'
    }
    page = requests.get(url, headers)
    tree = html.fromstring(page.content)
    tree.xpath("//h1/text()")
    table_rows = tree.xpath("//div[contains(@class, 'D(tbr)')]")
    assert len(table_rows) > 0
    parsed_rows = []
    for table_row in table_rows:
        parsed_row = []
        el = table_row.xpath("./div")
        none_count = 0
        for rs in el:
            try:
                (text,) = rs.xpath('.//span/text()[1]')
                parsed_row.append(text)
            except ValueError:
                parsed_row.append(np.NaN)
                none_count += 1
        if (none_count < 4):
            parsed_rows.append(parsed_row)
    df = pd.DataFrame(parsed_rows)
    df = df.set_index(0)
    df = df.transpose()
    cols = list(df.columns)
    cols[0] = 'Date'
    df = df.set_axis(cols, axis='columns', inplace=False)
    numeric_columns = list(df.columns)[1::]  # Take all columns, except the first (which is the 'Date' column)

    for column_name in numeric_columns:
        df[column_name] = df[column_name].str.replace(',', '')  # Remove the thousands separator
        df[column_name] = df[column_name].astype(np.float64)  # Convert the column to float64

    try:
        debt = (df['Total Debt'])
        debt = debt[1] * 1000 * factor
        if (math.isnan(debt)):
            return 0
        return debt
    except KeyError:
        return 0

#Calculates WACC of ticker
def getWACC(ticker, debt, dict):
    if (debt == 0):
        debt = costOfEquity(ticker)
    factor = 1000000
    market_cap = dict['shares'] * dict['mp'] * factor
    costEquity = costOfEquity(ticker)
    costDebt = costOfDebt(ticker, debt)
    WACC = (market_cap/(debt + market_cap)) * costEquity + (debt / (debt + market_cap)) * costDebt * (1 - .2)
    WACC /= 100
    WACC += 1
    return WACC

def getExchangeFactor(exchange):
    rate = 1.0
    if exchange == "KRW":
        rate = 0.00089
    elif exchange == "CNY":
        rate = 0.16
    elif exchange == "TWD":
        rate = 0.036
    elif exchange == "HKD":
        rate = 0.13
    return rate


def main():
    try:
        obj = realDCF()
        obj.dcf()
    except ConnectionError:
        print("Connection errors.")

if __name__ == "__main__":
    main()