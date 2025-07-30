import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from apscheduler.schedulers.blocking import BlockingScheduler
import logging
from datetime import datetime, timedelta
import time
import json
import plotly.graph_objects as go
import os

logging.basicConfig(
    filename="synthetics.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

login = 31891088
password = "@Tag0T5ima"
server = "Deriv-Demo"

symbols_list = [
    "Volatility 10 Index", "Volatility 25 Index", "Volatility 50 Index", "Volatility 100 Index",
    "Volatility 10 (1s) Index", "Volatility 15 (1s) Index", "Volatility 25 (1s) Index",
    "Volatility 30 (1s) Index", "Volatility 50 (1s) Index", "Volatility 75 (1s) Index",
    "Volatility 90 (1s) Index", "Volatility 100 (1s) Index", "Volatility 150 (1s) Index", "Volatility 250 (1s) Index"
]

symbols = dict.fromkeys(symbols_list, [])

if not os.path.exists("O1C.json"):
    with open("O1C.json", "w") as f:
        json.dump(symbols, f, indent=4)


mt5.initialize()#"C:\\Program Files\\MetaTrader 5 Terminal - 4\\terminal64.exe")


mt5.login(login, password, server)


def visualize_entry(df, pattern):
    pattern_df = pd.DataFrame(pattern, columns=["price", "index"])

    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['open'],
                                        high=df['high'],
                                        low=df['low'],
                                        close=df['close'],
                                        name='Price Action')],
                                        layout=go.Layout(
                                        width=1200,
                                        height=800))
    
    fig.add_trace(go.Scatter(
        x=pattern_df["index"],
        y=pattern_df['price'],
        mode='markers+text',
        marker=dict(size=10, color="blue"),
        text="Swing",
        textposition="top right",
        name="Markers"
    ))
    
    fig.write_image(f"{pattern[4][0]} pattern.jpeg")



def visualize_entry(df, pattern):
    pattern_df = pd.DataFrame(pattern, columns=["price", "index"])

    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['open'],
                                        high=df['high'],
                                        low=df['low'],
                                        close=df['close'],
                                        name='Price Action')],
                                        layout=go.Layout(
                                        width=1200,
                                        height=800))
    
    fig.add_trace(go.Scatter(
        x=pattern_df["index"],
        y=pattern_df['price'],
        mode='markers+text',
        marker=dict(size=10, color="blue"),
        text="Swing",
        textposition="top right",
        name="Markers"
    ))
    
    fig.write_image(f"{pattern[4][0]} pattern.jpeg")


def is_swing_low(lows):
    return (
        lows[3] <= lows[2] and
        lows[3] <  lows[1] and
        lows[3] <  lows[0] and
        lows[3] <= lows[4] and
        lows[3] <  lows[5] and
        lows[3] <  lows[6]
    )

def is_swing_high(highs):
    return (
        highs[3] >= highs[2] and
        highs[3] >  highs[1] and
        highs[3] >  highs[0] and
        highs[3] >= highs[4] and
        highs[3] >  highs[5] and
        highs[3] >  highs[6]
    )


def compute_atr(df, period=14):
    prev_close = df["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum(np.abs(df["high"] - prev_close),
                               np.abs(df["low"] - prev_close)))
    return tr.rolling(window=period).mean()

def find_swings(df, major=False, atr_multiplier=0.2, atr_period=5):
    atr_series = compute_atr(df, period=atr_period)
    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    
    lex = 0
    last_extreme_price = closes[0]
    direction = None

    swing_highs = []
    swing_lows = []

    n = len(df)

    for i in range(atr_period + 1, n):
        if i+3 >= n:
            if not major:
                if direction == "up":
                    swing_highs.append(("H", round(df.iloc[lex]["high"], 5), lex))
                    break
                elif direction == "down":
                    swing_lows.append(("L", round(df.iloc[lex]["low"], 5), lex))
                    break
                else:
                    break
            else:
                break
                
        atr = atr_series.iloc[i]
        if pd.isna(atr):
            continue

        price = closes[i]
        threshold = atr * atr_multiplier  # deviation in absolute terms

        if direction is None:
            if price - last_extreme_price > threshold:
                
                highs_window = highs[i-3:i+4]

                if is_swing_high(highs_window):                    
                    swing_highs.append(("H", df.iloc[i]["high"], i))
                    direction = 'down'
                    last_extreme_price = price
                    lex = i
                    
            elif last_extreme_price - price > threshold:
                
                lows_window = lows[i-3:i+4]

                if is_swing_low(lows_window):
                    swing_lows.append(("L", df.iloc[i]["low"], i))
                    direction = 'up'
                    last_extreme_price = price
                    lex = i

        elif direction == 'up':
            if price > last_extreme_price:
                
                highs_window = highs[i-3:i+4]

                if is_swing_high(highs_window):
                    last_extreme_price = price
                    lex = i
                    
            elif last_extreme_price - price > threshold:               

                lows_window = lows[i-3:i+4]

                if is_swing_low(lows_window):
                    
                    swing_highs.append(("H", round(df.iloc[lex]["high"], 5), lex)) if is_swing_high(highs[lex-3:lex+4]) else swing_lows.append(("L", round(df.iloc[lex]["low"], 5), lex))
                    direction = "down"
                    last_extreme_price = price
                    lex = i
            
            
        elif direction == 'down':
            if price < last_extreme_price:

                lows_window = lows[i-3:i+4]

                if is_swing_low(lows_window):
                    last_extreme_price = price
                    lex = i
                        
            elif price - last_extreme_price > threshold:

                highs_window = highs[i-3:i+4]

                if is_swing_high(highs_window):                        
                    swing_lows.append(("L", round(df.iloc[lex]["low"], 5), lex)) if is_swing_low(lows[lex-3:lex+4]) else swing_highs.append(("H", round(df.iloc[lex]["high"], 5), lex))
                    direction = 'up'
                    last_extreme_price = price
                    lex = i

    return swing_highs, swing_lows


def type1_buy_check(chosen, df) -> bool:
    prices, idxs = zip(*chosen)
    if df["low"].iloc[idxs[2]+4 : idxs[4]+4].min() <= prices[2]:
        return False
    if df["low"].iloc[idxs[4]+4:].min() <= prices[4]:
        return False
    if prices[4] > (prices[2] + prices[5]) / 2:
        return False
    return (
        prices[3] > prices[1] and
        prices[2] < prices[0] and
        prices[2] < prices[4] and
        prices[5] > prices[3]
    )

def type1_sell_check(chosen, df) -> bool:
    prices, idxs = zip(*chosen)
    if df["high"].iloc[idxs[2]+4 : idxs[4]+4].max() >= prices[2]:
        return False
    if df["high"].iloc[idxs[4]+4:].max() >= prices[4]:
        return False
    if prices[4] < (prices[2] + prices[5]) / 2:
        return False
    return (
        prices[3] < prices[1] and
        prices[2] > prices[0] and
        prices[2] > prices[4] and
        prices[5] < prices[3]
    )

def search_patterns(events, df, pattern, tag, idxs, tag_dict, idx_start=0, chosen=None):
    
    if chosen is None:
        chosen = []
    if len(chosen) == 6:
        if tag_dict[tag][0](chosen, df):
            yield chosen
        return

    pos = len(chosen)
    needed = pattern[pos]
    
    for j in range(idx_start, len(events)):
        t, p, i = events[j]
        if t != needed:
            continue
        if i in idxs[pos:pos+2]:
            if tag_dict[tag][1](pos, p, chosen):
                continue
            chosen.append((p,i))
            yield from search_patterns(events, df, pattern, tag, idxs, tag_dict, j+1, chosen)
        if chosen:
            chosen.pop()



def type1_buy_prune_function(position, price, existing):
    if position == 2:
        return price >= existing[0][0]

    if position == 4:
        return price <= existing[2][0]

def type1_sell_prune_function(position, price, existing):
    if position == 2:
        return price <= existing[0][0]

    if position == 4:
        return price >= existing[2][0]



def retry(func, *args, **kwargs):
    """
    Retries a function until it returns a non-None result or max_attempts is reached.
    """
    max_attempts = 3
    delay = 2
    
    for attempt in range(1, max_attempts + 1):
        result = func(*args, **kwargs)
        if result is not None:
            return result
        logging.warning(f"{func.__name__} returned None (attempt: {attempt}). \n Trying again in {delay} seconds")
        time.sleep(delay)
    logging.error(f"{func.__name__} failed after {max_attempts} attempts")
    return None

def initialize_mt5():
    if mt5.initialize():
        logging.info("MT5 initialized successfully")
        return True
    else:
        return None


def get_timeframe_start(timeframe):
    now = datetime.now()

    if timeframe == mt5.TIMEFRAME_M15:
        start = now - timedelta(days=now.weekday())
        return start.replace(hour=0, minute=0, second=0, microsecond=0)

    elif timeframe == mt5.TIMEFRAME_H4:
        month = now.month - 1
        return now.replace(month=month, day=1, hour=0, minute=0, second=0, microsecond=0)

    elif timeframe == mt5.TIMEFRAME_D1:
        month = ((now.month - 1) // 3) * 3 + 1
        return now.replace(month=month, day=1, hour=0, minute=0, second=0, microsecond=0)

    else:
        # Fallback: use past N bars for unsupported timeframes
        return now - timedelta(days=3)

def get_(symbol, timeframe):
    
    from_time = get_timeframe_start(timeframe)
    to_time = datetime.now()

    rates = mt5.copy_rates_range(symbol, timeframe, from_time, to_time)
    if rates is None:
        logging.error(f"Failed to get rates for {symbol}: {mt5.last_error()}")
        return

    rates_df = pd.DataFrame(rates)
        
    symbol_info = mt5.symbol_info(symbol)
    
    if symbol_info is None:
        logging.error(f"Failed to get symbol info for {symbol}: {mt5.last_error()}")
        return
        
    spread = symbol_info.spread
    logging.info(f"Spread for {symbol}: {spread}")
        

    return rates_df, symbol_info, spread * symbol_info.point

def place_order(order_type, sl, tp, price, lot, symbol): 

    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 100000,
        "comment": "FVG Limit Order",
        "type_time": mt5.ORDER_TIME_DAY,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Order failed, retcode={result.retcode}")
    else:
        logging.info(f"Order placed successfully $$$$$$$$$$$$$$")



def run_model_for_symbols(model_func, symbols, timeframe, function_dict):
    with ThreadPoolExecutor(max_workers=len(symbols)) as executor:
        futures = {executor.submit(model_func, symbol, timeframe, function_dict): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            logging.info(f"Finished checking {symbol}")
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error in {model_func.__name__} for {symbol}: {e}")
    


def chuks_type1_buy_model(symbol, timeframe, function_dict, tag="1B"):

    result = retry(get_, symbol, timeframe)    
    if not result:
        return
    df, symbol_info, spread = result
    
    highs, lows = find_swings(df)

    all_swings = highs + lows
    all_swings.sort(key=lambda x: x[2])
    indexes = [swing[2] for swing in all_swings]
    
    type1_buy_pattern = ["L", "H", "L", "H", "L", "H"]
    
    for pattern in search_patterns(all_swings, df, type1_buy_pattern, tag, indexes, function_dict):

        pattern_as_lists = [list(p) for p in pattern]

        # Load file
        with open("O1C.json", "r") as f:
            symbols = json.load(f)
        
        # Check and store if not present
        if pattern_as_lists not in symbols[symbol]:
            symbols[symbol].append(pattern_as_lists)
            
            with open("O1C.json", "w") as f:
                json.dump(symbols, f, indent=4)
                
            prices = [p[0] for p in pattern]
            indices = [p[1] for p in pattern]
            
            entry_price = min(df.iloc[indices[0]-1 : indices[0]+2]["high"].min(), prices[4])
            
            
            rd = 4
    
            round(entry_price, rd)
                
            atr = compute_atr(df).iloc[-1].round(rd)
            
            order = mt5.ORDER_TYPE_BUY_LIMIT
            
            risk = 200.35 * 0.1  # 10% of account || should be 10% of starting capital
            sl = round(prices[2] - (1.5 * atr) + spread, rd)  # Protected Low
            sl_pips = round(abs(entry_price - sl), rd)
            tp = max(round(entry_price + (3 * sl_pips) + spread, rd), round(prices[5] + spread, rd))
            lot = max(round((risk/sl_pips) * 0.0001, 2), symbol_info.volume_min)
            
            place_order(order, sl, tp, entry_price, lot, symbol)

            try:
                visualize_entry(df, pattern)
            except Exception as e:
                logging.error(f"Error while plotting entry: {e}")



def chuks_type1_sell_model(symbol, timeframe, function_dict, tag="1S"):
    
    result = retry(get_, symbol, timeframe)
    if not result:
        return
    df, symbol_info, spread = result
    
    highs, lows = find_swings(df)
   
    all_swings = highs + lows
    all_swings.sort(key=lambda x: x[2])
    indexes = [swing[2] for swing in all_swings]
    
    type1_sell_pattern = ["H", "L", "H", "L", "H", "L"]
    
    for pattern in search_patterns(all_swings, df, type1_sell_pattern, tag, indexes, function_dict):

        pattern_as_lists = [list(p) for p in pattern]

        # Load file
        with open("O1C.json", "r") as f:
            symbols = json.load(f)
        
        # Check and store if not present
        if pattern_as_lists not in symbols[symbol]:
            symbols[symbol].append(pattern_as_lists)
            
            with open("O1C.json", "w") as f:
                json.dump(symbols, f, indent=4)
                
            prices = [p[0] for p in pattern]
            indices = [p[1] for p in pattern]
            
            entry_price = max(df.iloc[indices[0]-1 : indices[0]+2]["low"].max(), prices[4])
            
            
            rd = 4
    
            round(entry_price, rd)
                
            atr = compute_atr(df).iloc[-1].round(rd)
            
            risk = 200.35 * 0.1  # 10% of account   
            sl = round(prices[2] + (1.5 * atr) - spread, rd)  # atr
            sl_pips = round(abs(entry_price - sl), rd)
            tp = min(round(entry_price - (3 * sl_pips) - spread, rd), round(prices[5] - spread, rd))
            lot = max(round((risk/sl_pips) * 0.0001, 2), symbol_info.volume_min)
            
            place_order(order, sl, tp, entry_price, lot, symbol)

            try:
                visualize_entry(df, pattern)
            except Exception as e:
                logging.error(f"Error while plotting entry: {e}")


def trading_job():
    """Main trading function"""
    if not retry(initialize_mt5):
        return

    if mt5.positions_total() >= 10:  # Max 10 positions at once to follow 10 trades rule
        logging.info(f"Maximum positions reached.")
        return
        
    # Set the timeframe
    timeframes = [mt5.TIMEFRAME_M15, mt5.TIMEFRAME_M5]
    #higher_timeframe = mt5.TIMEFRAME_H4
    
    symbols = ["Volatility 10 Index", "Volatility 25 Index", "Volatility 50 Index", "Volatility 100 Index",
               "Volatility 10 (1s) Index", "Volatility 15 (1s) Index", "Volatility 25 (1s) Index",
               "Volatility 30 (1s) Index", "Volatility 50 (1s) Index", "Volatility 75 (1s) Index",
               "Volatility 90 (1s) Index", "Volatility 100 (1s) Index", "Volatility 150 (1s) Index", "Volatility 250 (1s) Index"]
    
    #premium_symbols, discount_symbols = sort_symbols(symbols, higher_timeframe)

    model_functions_dict = {
        "1B":[type1_buy_check, type1_buy_prune_function],
        "1S":[type1_sell_check, type1_sell_prune_function]
    }

    try:
        for tf in timeframes:
            logging.info(f"Looking for type1 buy setups on {tf}")
            run_model_for_symbols(chuks_type1_buy_model, symbols, tf, model_functions_dict)
            logging.info(f"Looking for type1 sell setups on {tf}")
            run_model_for_symbols(chuks_type1_sell_model, symbols, tf, model_functions_dict)
    finally:
        mt5.shutdown()
        logging.info("MT5 connection closed")
        logging.info("Trading bot stopped")


scheduler = BlockingScheduler()
scheduler.add_job(
    trading_job, 
    'cron', 
    day_of_week='mon-sun', 
    #hour='1-19/2',
    hour="*",
    minute="15,30,45,0",
    timezone='Africa/Lagos'
)
try:
    scheduler.start()
except (KeyboardInterrupt, SystemExit):
    pass