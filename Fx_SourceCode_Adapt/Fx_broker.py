import logging
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import os
import pickle
from Fx_SourceCode_Adapt.Fx_data_module import get_fx_data  # Import the data functions for FX
from Fx_SourceCode_Adapt.Fx_utils import generate_random_name  # If any other utils are needed for FX data
from numba import jit 

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class Position:
    """Position object to track the FX pair's position details"""
    pair: str      # Currency pair (e.g., EURUSD=X)
    quantity: int  # Position size
    entry_price: float  # Entry price for the FX pair
    
class StopLoss:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def check_loss(self, position: Position, current_price: float):
        if (position.entry_price - current_price) / position.entry_price >= self.threshold:
            return True  # Trigger stop loss
        return False    

@dataclass
class Broker:
    """Broker class adapted for FX backtesting"""
    cash: float 
    positions: dict = None
    transaction_log: pd.DataFrame = None
    entry_prices: dict = None
    verbose: bool = True

    def initialize_blockchain(self, name: str):
        """Initializes blockchain to store the backtest results."""
        if not os.path.exists('blockchain'):
            os.makedirs('blockchain')
        chains = os.listdir('blockchain')
        ending = f'{name}.pkl'
        if ending in chains:
            if self.verbose:
                logging.warning(f"Blockchain with name {name} already exists. Please use a different name.")
            with open(f'blockchain/{name}.pkl', 'rb') as f:
                self.blockchain = pickle.load(f)
            return

        self.blockchain = Blockchain(name)
        self.blockchain.store()

        if self.verbose:
            logging.info(f"Blockchain with name {name} initialized and stored.")

    def __post_init__(self):
        """Post initialization to set up the positions and transaction log"""
        if self.positions is None:
            self.positions = {}
        if self.transaction_log is None:
            self.transaction_log = pd.DataFrame(columns=['Date', 'Action', 'Pair', 'Quantity', 'Price', 'Cash'])
        if self.entry_prices is None:
            self.entry_prices = {}

    def buy(self, pair: str, quantity: int, price: float, date: datetime):
        """Buy a currency pair in the portfolio"""
        total_cost = price * quantity
        if self.cash >= total_cost:
            self.cash -= total_cost
            if pair in self.positions:
                position = self.positions[pair]
                new_quantity = position.quantity + quantity
                new_entry_price = ((position.entry_price * position.quantity) + (price * quantity)) / new_quantity
                position.quantity = new_quantity
                position.entry_price = new_entry_price
            else:
                self.positions[pair] = Position(pair, quantity, price)
            self.log_transaction(date, 'BUY', pair, quantity, price)
            self.entry_prices[pair] = price
        else:
            if self.verbose:
                logging.warning(f"Not enough cash to buy {quantity} units of {pair} at {price}. Available cash: {self.cash}")

    def sell(self, pair: str, quantity: int, price: float, date: datetime):
        """Sell a currency pair in the portfolio"""
        if pair in self.positions and self.positions[pair].quantity >= quantity:
            position = self.positions[pair]
            position.quantity -= quantity
            self.cash += price * quantity

            if position.quantity == 0:
                del self.positions[pair]
                del self.entry_prices[pair]
            self.log_transaction(date, 'SELL', pair, quantity, price)
        else:
            if self.verbose:
                logging.warning(f"Not enough units of {pair} to sell {quantity}. Position size: {self.positions.get(pair, 0)}")

    def log_transaction(self, date, action, pair, quantity, price):
        """Log every transaction to the transaction log"""
        transaction = pd.DataFrame([{
            'Date': date,
            'Action': action,
            'Pair': pair,
            'Quantity': quantity,
            'Price': price,
            'Cash': self.cash
        }])

        self.transaction_log = pd.concat([self.transaction_log, transaction], ignore_index=True)

    def get_cash_balance(self):
        """Get the current cash balance"""
        return self.cash

    def get_transaction_log(self):
        """Return the transaction log"""
        return self.transaction_log

    def get_portfolio_value(self, market_prices: dict):
        """Calculate the current value of the portfolio in terms of cash and open positions"""
        portfolio_value = self.cash
        for pair, position in self.positions.items():
            portfolio_value += position.quantity * market_prices.get(pair, 0)
        return portfolio_value

    def execute_portfolio(self, portfolio: dict, prices: dict, date: datetime):
        """Execute the trades in the portfolio based on weights and available cash"""
        # First handle sell orders
        for pair, weight in portfolio.items():
            price = prices.get(pair)
            if price is None:
                if self.verbose:
                    logging.warning(f"Price for {pair} not available on {date}")
                continue
            
            total_value = self.get_portfolio_value(prices)
            target_value = total_value * weight
            current_value = self.positions.get(pair, Position(pair, 0, 0)).quantity * price
            diff_value = target_value - current_value
            quantity_to_trade = int(diff_value / price)

            if quantity_to_trade < 0:
                self.sell(pair, abs(quantity_to_trade), price, date)

        # Then handle buy orders
        for pair, weight in portfolio.items():
            price = prices.get(pair)
            if price is None:
                if self.verbose:
                    logging.warning(f"Price for {pair} not available on {date}")
                continue

            total_value = self.get_portfolio_value(prices)
            target_value = total_value * weight
            current_value = self.positions.get(pair, Position(pair, 0, 0)).quantity * price
            diff_value = target_value - current_value
            quantity_to_trade = int(diff_value / price)

            if quantity_to_trade > 0:
                available_cash = self.get_cash_balance()
                cost = quantity_to_trade * price
                
                if cost <= available_cash:
                    self.buy(pair, quantity_to_trade, price, date)
                else:
                    if self.verbose:
                        logging.warning(f"Not enough cash to buy {quantity_to_trade} of {pair} on {date}. Needed: {cost}, Available: {available_cash}")
                        logging.info(f"Buying as many units of {pair} as possible with available cash.")
                    quantity_to_trade = int(available_cash / price)
                    self.buy(pair, quantity_to_trade, price, date)

    def get_transaction_log(self):
        """Returns the transaction log"""
        return self.transaction_log


