# fetch_financial_data.py
# pip install yfinance nasdaq-data-link intrinio-sdk alpha_vantage pycoingecko pandas

import pandas as pd
import yfinance as yf
import nasdaqdatalink
import intrinio_sdk as intrinio
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from pycoingecko import CoinGeckoAPI
from datetime import datetime

#import config


def get_data_yahoo_finance(ticker, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance.
    We can download information for multiple tickers at once by providing a list of tickers (["AAPL", "MSFT"]) or multiple tickers as a string ("AAPL MSFT").
    We can set auto_adjust=True to download only the adjusted prices.
    We can additionally download dividends and stock splits by setting actions='inline'. Those actions can also be used to manually adjust the prices or for other analyses.
    Specifying progress=False disables the progress bar.
    The interval argument can be used to download data in different frequencies. We could also download intraday data as long as the requested period is shorter than 60 days.
    """
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return df
'''
def get_data_nasdaq_datalink(ticker, start_date, end_date, api_key):
    """
    Fetch historical stock data from Nasdaq Data Link.
    """
    # Authenticate
    nasdaqdatalink.ApiConfig.api_key = api_key
    # Download data
    df = nasdaqdatalink.get(dataset=f"WIKI/{ticker}",
                            start_date=start_date,
                            end_date=end_date)
    return df

def get_data_intrinio(ticker, start_date, end_date, api_key):
    """
    Fetch historical stock data from Intrinio.
    """
    # Authenticate
    intrinio.ApiClient().set_api_key(api_key)
    security_api = intrinio.SecurityApi()
    # Request data
    r = security_api.get_security_stock_prices(
        identifier=ticker,
        start_date=start_date,
        end_date=end_date,
        frequency="daily",
        page_size=10000
    )
    df = (
        pd.DataFrame(r.stock_prices_dict)
        .sort_values("date")
        .set_index("date")
    )
    return df

def get_data_alpha_vantage_stock(ticker, api_key):
    """
    Fetch daily stock data from Alpha Vantage.
    """
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
    data.sort_index(inplace=True)
    return data

def get_data_alpha_vantage_crypto(symbol, market, api_key):
    """
    Fetch daily cryptocurrency data from Alpha Vantage.
    """
    cc = CryptoCurrencies(key=api_key, output_format='pandas')
    data, meta_data = cc.get_digital_currency_daily(symbol=symbol, market=market)
    data.sort_index(inplace=True)
    return data

def get_data_coingecko(coin_id, vs_currency, days):
    """
    Fetch OHLC cryptocurrency data from CoinGecko.
    """
    cg = CoinGeckoAPI()
    ohlc = cg.get_coin_ohlc_by_id(id=coin_id, vs_currency=vs_currency, days=days)
    ohlc_df = pd.DataFrame(ohlc)
    ohlc_df.columns = ["date", "open", "high", "low", "close"]
    ohlc_df["date"] = pd.to_datetime(ohlc_df["date"], unit="ms")
    return ohlc_df
'''
def main():
    # Set API keys here
    #NASDAQ_API_KEY = config.NASDAQ_API_KEY
    #INTRINIO_API_KEY = config.INTRINIO_API_KEY
    #ALPHA_VANTAGE_API_KEY = config.ALPHA_VANTAGE_API_KEY

    # Define the parameters
    ticker = "AAPL"
    start_date = "2011-01-01"
    end_date = "2021-12-31"

    # Yahoo Finance
    print("Fetching data from Yahoo Finance...")
    df_yahoo = get_data_yahoo_finance(ticker, start_date, end_date)
    print(f"Yahoo Finance data:\n{df_yahoo.head()}\n")
    '''
    # Nasdaq Data Link
    print("Fetching data from Nasdaq Data Link...")
    df_nasdaq = get_data_nasdaq_datalink(ticker, start_date, end_date, NASDAQ_API_KEY)
    print(f"Nasdaq Data Link data:\n{df_nasdaq.head()}\n")

    # Intrinio
    print("Fetching data from Intrinio...")
    df_intrinio = get_data_intrinio(ticker, start_date, end_date, INTRINIO_API_KEY)
    print(f"Intrinio data:\n{df_intrinio.head()}\n")

    # Alpha Vantage Stock Data
    print("Fetching stock data from Alpha Vantage...")
    df_alpha_stock = get_data_alpha_vantage_stock(ticker, ALPHA_VANTAGE_API_KEY)
    print(f"Alpha Vantage Stock data:\n{df_alpha_stock.head()}\n")

    # Alpha Vantage Crypto Data
    print("Fetching cryptocurrency data from Alpha Vantage...")
    crypto_symbol = "BTC"
    market = "USD"
    df_alpha_crypto = get_data_alpha_vantage_crypto(crypto_symbol, market, ALPHA_VANTAGE_API_KEY)
    print(f"Alpha Vantage Crypto data:\n{df_alpha_crypto.head()}\n")

    # CoinGecko
    print("Fetching data from CoinGecko...")
    coin_id = "bitcoin"
    vs_currency = "usd"
    days = 14
    df_coingecko = get_data_coingecko(coin_id, vs_currency, days)
    print(f"CoinGecko data:\n{df_coingecko.head()}\n")
    '''
if __name__ == "__main__":
    main()

