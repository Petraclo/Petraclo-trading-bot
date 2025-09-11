import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Tuple, Dict, Optional
import logging
import time
import json
import os
import plotly.graph_objects as go

trades_file = "PTC-v2.json"

logging.basicConfig(
    filename="PTC-v2.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

#===FILES===
def load_symbols_cache(filename=trades_file):
    with open(filename, "r") as f:
        data = json.load(f)
    # Convert lists back to sets
    return {symbol: set(tuple(p) for p in patterns) for symbol, patterns in data.items()}

def save_symbols_cache(symbols_cache, filename=trades_file):
    with open(filename, "w") as f:
        # Convert sets to lists for JSON serialization
        json.dump({symbol: [list(p) for p in patterns] for symbol, patterns in symbols_cache.items()}, f, indent=4)


if not os.path.exists(trades_file):
    symbols_list = [
    "Volatility 10 Index", "Volatility 25 Index", "Volatility 50 Index", "Volatility 75 Index", "Volatility 100 Index",
    "Volatility 10 (1s) Index", "Volatility 15 (1s) Index", "Volatility 25 (1s) Index",
    "Volatility 30 (1s) Index", "Volatility 50 (1s) Index", "Volatility 75 (1s) Index",
    "Volatility 90 (1s) Index", "Volatility 100 (1s) Index", "Volatility 150 (1s) Index", "Volatility 250 (1s) Index"
    ]

    symbols = {s: set() for s in symbols_list}
    save_symbols_cache(symbols)
        
#=======HELPER FUNCTIONS========

        
#===VISUALISATION===
def visualize_entry(df, pattern: List[Tuple], symbol: str, tf: int, order_type: int, entry: float) -> None :
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

    order = "buy setup" if order_type == 2 else "sell setup"
    fig.write_image(f"{symbol} {order} at {entry} on {tf} minute timeframe.jpeg")


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

def find_swings(df, dp, major=False, atr_multiplier=0.2, atr_period=5):
    df = df.round(dp)
    
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
                    swing_highs.append(("H", df.iloc[lex]["high"], lex))
                    break
                elif direction == "down":
                    swing_lows.append(("L", df.iloc[lex]["low"], lex))
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
                    
                    swing_highs.append(("H", df.iloc[lex]["high"], lex)) if is_swing_high(highs[lex-3:lex+4]) else swing_lows.append(("L", df.iloc[lex]["low"], lex))
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
                    swing_lows.append(("L", df.iloc[lex]["low"], lex)) if is_swing_low(lows[lex-3:lex+4]) else swing_highs.append(("H", df.iloc[lex]["high"], lex))
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
    if prices[4] > (prices[1] + prices[2]) / 2:
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
    if prices[4] < (prices[1] + prices[2]) / 2:
        return False
    return (
        prices[3] < prices[1] and
        prices[2] > prices[0] and
        prices[2] > prices[4] and
        prices[5] < prices[3]
    )

def search_patterns(events, df, pattern, idxs, idx_map, tag, tag_dict, idx_start=0, prev_idx=None, chosen=None):
    if chosen is None:
        chosen = []

    # Base case
    if len(chosen) == len(pattern):
        # Function that checks legitimate patterns depending on the tag
        if tag_dict[tag][0](chosen, df):
            yield chosen
        return

    pos = len(chosen)
    needed = pattern[pos]

    for j in range(idx_start, len(events)):
        t, p, i = events[j]

        if t != needed:
            continue
            
        if prev_idx is None:
            # First match: accept any match, set its index
            if tag_dict[tag][1](pos, p, chosen):
                continue
            current_idx = idx_map.get(i)
            chosen.append((p,i))
            yield from search_patterns(events, df, pattern, idxs, idx_map, tag, tag_dict, j+1, current_idx, chosen)
            chosen.pop()

        else:
            # Subsequent matches: check adjacency constraint
            if i in idxs[prev_idx+1:prev_idx+3]:
                if tag_dict[tag][1](pos, p, chosen):
                    continue
                current_idx = idx_map.get(i)
                chosen.append((p,i))
                yield from search_patterns(events, df, pattern, idxs, idx_map, tag, tag_dict, j+1, current_idx, chosen)
                chosen.pop()
                
            else:
                return


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

def login_mt5(login, password, server):
    if mt5.login(login, password, server):
        logging.info(f"Successfully logged into {login}")
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

    return rates_df, symbol_info, spread * symbol_info.trade_tick_size

def place_order(order_type, sl, tp, price, lot, symbol): 

    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 2000,
        "magic": 100000,
        "comment": "FVG Limit Order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    if mt5.orders_total() < 15:
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Order failed for {symbol}, retcode={result.retcode}")
        else:
            logging.info(f"Order placed successfully on {symbol} $$$$$$$$$$$$$$")
    else:
        logging.info("Max number of orders placed")

def sort_symbols(symbols_list, htf):
    premium_symbols = []
    discount_symbols = []
    
    for symbol in symbols_list:
        
        result = retry(get_, symbol, htf)
        if not result:
            continue
        df, symbol_info, _ = result
        
        highs, lows = find_swings(df, major=True, atr_multiplier=2.0, atr_period=7)
        last_high, last_low = highs[-1][1], lows[-1][1]

        premium_level = (last_high + last_low)/ 2

        if symbol_info.bid >= premium_level:
            premium_symbols.append(symbol)
        else:
            discount_symbols.append(symbol)

    return premium_symbols, discount_symbols

def run_model_for_symbols(model_func, symbols, timeframe, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=len(symbols)) as executor:
        futures = {executor.submit(model_func, symbol, timeframe, *args, **kwargs): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error in {model_func.__name__} for {symbol}: {e}")
                
    
def chuks_type1_model(symbol, timeframe, symbols_cache, function_dict, tag):
    result = retry(get_, symbol, timeframe)    
    if not result:
        return
    df, symbol_info, spread = result

    tick_size = symbol_info.trade_tick_size 
    rd = min(abs(Decimal(str(tick_size)).as_tuple().exponent), symbol_info.digits)

    highs, lows = find_swings(df, rd)
    all_swings = highs + lows
    all_swings.sort(key=lambda x: x[2])
    indexes = [swing[2] for swing in all_swings]
    index_map = {index: i for i, index in enumerate(indexes)}

    # Set buy/sell parameters
    if tag == "1B":
        pattern = ["L", "H", "L", "H", "L", "H"]
        order_type = mt5.ORDER_TYPE_BUY_LIMIT
        entry_price_func = lambda df, idxs, prices: min(df.iloc[idxs[0]-1 : idxs[0]+2]["high"].min(), prices[4])
        sl_func = lambda prices, atr, spread: prices[2] - (1.5 * atr) + spread
        tp_func = lambda entry, sl_ticks, spread, prices: max(entry + (3 * sl_ticks) + spread, prices[5] + spread)
    else:
        pattern = ["H", "L", "H", "L", "H", "L"]
        order_type = mt5.ORDER_TYPE_SELL_LIMIT
        entry_price_func = lambda df, idxs, prices: max(df.iloc[idxs[0]-1 : idxs[0]+2]["low"].max(), prices[4])
        sl_func = lambda prices, atr, spread: prices[2] + (1.5 * atr) - spread
        tp_func = lambda entry, sl_ticks, spread, prices: min(entry - (3 * sl_ticks) - spread, prices[5] - spread)

    for pattern_match in search_patterns(all_swings, df, pattern, indexes, index_map, tag, function_dict):
        prices = tuple([p[0] for p in pattern_match])

        if prices not in symbols_cache[symbol]:
            logging.info(f"Present pattern on {symbol}: {prices}")
            symbols_cache[symbol].add(prices)

            indices = [p[1] for p in pattern_match]
            entry_price = entry_price_func(df, indices, prices)
            atr = compute_atr(df).iloc[-1].round(rd)

            risk = round(84.70 * 0.1, 1)  # TODO: link to starting capital
            
            sl = round(sl_func(prices, atr, spread), rd)

            sl_ticks = round(abs(entry_price - sl), rd)
            n_ticks = round(sl_ticks / tick_size, 1)
            tick_value = symbol_info.trade_tick_value

            lot_precision = abs(Decimal(str(symbol_info.volume_step)).as_tuple().exponent)
            lot = round(risk / (tick_value * n_ticks), lot_precision)
            lot = max(symbol_info.volume_min, min(lot, symbol_info.volume_max))
            actual_risk = round(lot * tick_value * n_ticks, 1)

            if actual_risk < risk:
                sl_ticks = (risk * tick_size) / (lot * tick_value)
                sl = round(entry_price - sl_ticks if tag == "1B" else entry_price + sl_ticks, rd)
            elif actual_risk > risk:
                logging.info(f"Risk too high for this trade: {actual_risk}")
                continue

            tp = round(tp_func(entry_price, sl_ticks, spread, prices), rd)

            try:
                visualize_entry(df, pattern_match, symbol, timeframe, order_type, entry_price)
            except Exception as e:
                logging.error(f"Error while plotting entry: {e}")

            place_order(order_type, sl, tp, entry_price, lot, symbol)


def trading_job():
    """Main trading function"""
    if not retry(initialize_mt5):
        return

    
    login = 40396201
    password = "Ttobs_der1v"
    server = "Deriv-Demo"

    if not retry(login_mt5, login, password, server):
        return

    symbols_cache = load_symbols_cache()
    # Set the timeframe
    timeframes = [mt5.TIMEFRAME_M15, mt5.TIMEFRAME_M5]
    #higher_timeframe = mt5.TIMEFRAME_H4
    
    symbols = ["Volatility 10 Index", "Volatility 25 Index", "Volatility 75 Index", "Volatility 50 Index", "Volatility 100 Index",
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
            logging.info(f"Looking for type1 buy setups on {tf} minute")
            run_model_for_symbols(chuks_type1_model, symbols, tf, symbols_cache, model_functions_dict, tag="1B")
            logging.info(f"Looking for type1 sell setups on {tf} minute")
            run_model_for_symbols(chuks_type1_model, symbols, tf, symbols_cache, model_functions_dict, tag="1S")
    finally:
        save_symbols_cache(symbols_cache)
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