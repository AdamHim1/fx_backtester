import pandas as pd
import numpy as np
import logging
from datetime import datetime
from numba import jit
from dataclasses import dataclass


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FxBacktest:
    initial_date: datetime
    final_date: datetime
    universe: list
    broker: Broker = None
    initial_cash: int = 1000000  # Initial cash in the portfolio
    
    def __init__(self, initial_date: datetime, final_date: datetime, universe: list, broker: Broker,
                 information_class: object, risk_model: object, name_blockchain: str, verbose: bool):
        self.initial_date = initial_date
        self.final_date = final_date
        self.universe = universe
        self.broker = broker
        self.information_class = information_class  # Add this line
        self.risk_model = risk_model
        self.name_blockchain = name_blockchain
        self.verbose = verbose

    def run_backtest(self):
        # Get historical FX data for the universe
        logging.info("Retrieving FX data for the universe.")
        fx_data = get_fx_pairs_data(self.universe, self.initial_date.strftime('%Y-%m-%d'), self.final_date.strftime('%Y-%m-%d'))

        # Log the data to ensure it's loaded correctly
        logging.info(f"Data loaded for {', '.join(self.universe)}")

        # Process each date in the range
        for t in pd.date_range(start=self.initial_date, end=self.final_date, freq='D'):
            self._process_date(fx_data, t)
            
        logging.info(f"Backtest completed. Final portfolio value: {self.broker.get_portfolio_value(fx_data)}")
    
    def _process_date(self, fx_data, t: datetime):
        # Filter the data for the current day
        daily_data = fx_data[fx_data['Date'] == t]
        
        # Simulate buy/sell logic
        for row in daily_data.itertuples():
            ticker = row.ticker
            close_price = row['Close']
            
            # Example: Execute a buy/sell decision based on some logic (here, simply buying)
            self.broker.buy(ticker, quantity=100, price=close_price, date=t)

        # Log the transaction at the end of the day
        logging.info(f"Processed transactions for {t}. Current cash: {self.broker.get_cash_balance()}.")

def test_backtest():
    # Create the backtest with a list of FX pairs
    fx_pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]
    backtest = FxBacktest(initial_date=datetime(2021, 1, 1), final_date=datetime(2021, 12, 31), universe=fx_pairs)
    
    # Run the backtest
    backtest.run_backtest()

# Run the test
test_backtest()
