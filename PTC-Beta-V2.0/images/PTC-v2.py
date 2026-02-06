import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime, timedelta
from dateutil import parser
from decimal import Decimal
from typing import List, Tuple, Dict, Optional
from threading import Lock
from custom import MaxOrdersReached
from logging.handlers import RotatingFileHandler
import logging
import time
import json
import os
import plotly.graph_objects as go

EXPOSURE_LIMIT = 10
order_lock = Lock()

M5_trades_file = "M5_trades.json"
M15_trades_file = "M15_trades.json"

M5_symbols = []
    
M15_symbols = ["Volatility 100 (1s) Index"]

M5_htf_symbols = ["Volatility 10 Index", "Volatility 100 Index", "Volatility 100 (1s) Index"]

M15_htf_symbols = ["Volatility 10 Index", "Volatility 25 Index",
                "Volatility 50 Index", "Volatility 100 Index", "Volatility 10 (1s) Index"]

rotating_handler = RotatingFileHandler(
    filename="PTC-v2.log",
    maxBytes=1 * 1024 * 1024,
    backupCount=5
)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
rotating_handler.setFormatter(formatter)

logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(rotating_handler)


#===FILES===
def load_symbols_cache(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    # Convert lists back to sets
    return {symbol: set(tuple(p) for p in patterns) for symbol, patterns in data.items()}

def save_symbols_cache(symbols_cache, filename):
    with open(filename, "w") as f:
        # Convert sets to lists for JSON serialization
        json.dump({symbol: [list(p) for p in patterns] for symbol, patterns in symbols_cache.items()}, f, indent=4)

def clear_symbols_cache(filename):
    if os.path.exists(filename):
        os.remove(filename)

def create_symbols_cache(filename):
    if not os.path.exists(filename):
        if filename == "M5_trades.json":
            symbols_list = M5_htf_symbols + M5_symbols
        elif filename == "M15_trades.json":
            symbols_list = M15_htf_symbols + M15_symbols

        symbols = {s: set() for s in symbols_list}
        save_symbols_cache(symbols, filename)

def load_M5_symbol_status():
    modified = False
    clear = False
    with open("M5_symbol_status.json", "r") as f:
        data = json.load(f)

    for symbol, status_dict in data.items():
        if parser.parse(status_dict["reset_date"]) < datetime.now():
            status_dict["start_date"] =  str(datetime.now())
            status_dict["reset_date"] =  str(datetime.now() + timedelta(days=3))
            status_dict["status"] = "default"
            clear = True
            modified = True

    for symbol, status_dict in data.items():
        if status_dict["status"] == "default":
            deals = mt5.history_deals_get(
                parser.parse(status_dict["start_date"]),
                parser.parse(status_dict["reset_date"]),
                group=symbol)
            
            if not deals:
                continue
            else:
                deal = deals[-1]            
                # Only react to CLOSED losing trades
                if getattr(deal, "entry", None) == mt5.DEAL_ENTRY_OUT:
                    profit = getattr(deal, "profit", 0)
                    magic = getattr(deal, "magic", 0)
                    if profit < 0 and magic == 100000:
                        logging.info(f"Disabling trading on {symbol} on M5 due to losing trade")
                        status_dict["status"] = "disabled"
                        modified = True
                    elif profit > 0 and magic == 100000:
                        logging.info(f"Adapting trading on {symbol} on M5 due to winning trade")
                        status_dict["status"] = "adapted"
                        modified = True
                else:
                    logging.info("Order not closed yet")

    if modified:
        with open("M5_symbol_status.json", "w") as f:
            json.dump(data, f, indent=4)

    return data, clear


def load_M15_symbol_status():
    modified = False
    clear = False
    with open("M15_symbol_status.json", "r") as f:
        data = json.load(f)

    for symbol, status_dict in data.items():
        if parser.parse(status_dict["reset_date"]) < datetime.now():
            status_dict["start_date"] = str((datetime.now() - timedelta(days=datetime.now().weekday())).replace(hour=0, minute=0, second=0, microsecond=0))
            status_dict["reset_date"] = str(parser.parse(status_dict["start_date"]) + timedelta(days=7))
            status_dict["status"] = "default"
            clear = True
            modified = True

    for symbol, status_dict in data.items():
        if status_dict["status"] == "default":
            deals = mt5.history_deals_get(
                parser.parse(status_dict["start_date"]),
                parser.parse(status_dict["reset_date"]),
                group=symbol)
            
            if not deals:
                continue
            else:
                deal = deals[-1]
                # Only react to CLOSED losing trades
                if getattr(deal, "entry", None) == mt5.DEAL_ENTRY_OUT:
                    profit = getattr(deal, "profit", 0)
                    magic = getattr(deal, "magic", 0)
                    if profit < 0 and magic == 200000:
                        logging.info(f"Disabling trading on {symbol} on M15 due to losing trade")
                        status_dict["status"] = "disabled"
                        modified = True
                    elif profit > 0 and magic == 200000:
                        logging.info(f"Adapting trading on {symbol} on M15 due to winning trade")
                        status_dict["status"] = "adapted"
                        modified = True
                else:
                    logging.info("Order not closed yet")

    if modified:
        with open("M15_symbol_status.json", "w") as f:
            json.dump(data, f, indent=4)

    return data, clear

def load_order_params():
    with open("order_params.json", "r") as f:
        params = json.load(f)

    return params
        
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

def find_swings(df, dp=None, major=False, atr_multiplier=0.2, atr_period=5):
    if dp:
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

def search_patterns(events, df, pattern, idxs, idx_map, tag, tag_dict, sensitivity, idx_start=0, prev_idx=None, chosen=None):
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
            yield from search_patterns(events, df, pattern, idxs, idx_map, tag, tag_dict, sensitivity, j+1, current_idx, chosen)
            chosen.pop()

        else:
            # Subsequent matches: check adjacency constraint
            if i in idxs[prev_idx+1:prev_idx+(1+sensitivity)]:
                if tag_dict[tag][1](pos, p, chosen):
                    continue
                current_idx = idx_map.get(i)
                chosen.append((p,i))
                yield from search_patterns(events, df, pattern, idxs, idx_map, tag, tag_dict, sensitivity, j+1, current_idx, chosen)
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

    elif timeframe == mt5.TIMEFRAME_H1:
        start = now - timedelta(days=30)
        return start

    elif timeframe == mt5.TIMEFRAME_H4:
        month = now.month - 1 if now.month > 1 else 1
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
    
    acct_info = mt5.account_info()
    
    if acct_info is None:
        logging.error(f"Failed to get account info: {mt5.last_error()}")
        return
        
    balance = acct_info.balance

    return rates_df, symbol_info, spread * symbol_info.point, balance

def place_order(order_type, sl, tp, price, lot, symbol, tf):
    with order_lock:
        if mt5.orders_total() + mt5.positions_total() < EXPOSURE_LIMIT:
            placed = False 

            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": lot,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 2000,
                "magic": 100000 if tf == mt5.TIMEFRAME_M5 else 200000,
                "comment": "Chuks Type1",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }

            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"Order failed for {symbol}, retcode={result.retcode}")
            else:
                logging.info(f"Order placed successfully on {symbol} $$$$$$$$$$$$$$")
                placed = True        

            return placed
        
        else:
            raise MaxOrdersReached("Maximum number of orders/positions reached.")

def close_expired_orders_and_trades(symbol_status, tf_magic):
    """
    Remove:
      - pending orders older than symbol start_date
      - active positions opened before start_date

    Only affects trades with matching magic number.
    """
    orders = mt5.orders_get() or []

    for order in orders:

        if order.magic != tf_magic:
            continue

        symbol = order.symbol
        if symbol not in symbol_status:
            continue

        cutoff = parser.parse(symbol_status[symbol]["start_date"])
        order_time = datetime.fromtimestamp(order.time_setup)

        if order_time < cutoff:

            result = mt5.order_send({
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": order.ticket,
            })

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(
                    f"Removed expired PENDING on {symbol} | "
                    f"opened={order_time} | reset={cutoff}"
                )
            else:
                logging.info(f"Failed to close PENDING on {symbol} : {result.retcode}")

    positions = mt5.positions_get() or []

    for pos in positions:

        if pos.magic != tf_magic:
            continue

        symbol = pos.symbol
        if symbol not in symbol_status:
            continue

        cutoff = parser.parse(symbol_status[symbol]["start_date"])
        open_time = datetime.fromtimestamp(pos.time)

        if open_time < cutoff:

            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "position": pos.ticket,
                "type": (
                    mt5.ORDER_TYPE_SELL
                    if pos.type == mt5.POSITION_TYPE_BUY
                    else mt5.ORDER_TYPE_BUY
                ),
                "price": (
                    mt5.symbol_info_tick(symbol).bid
                    if pos.type == mt5.POSITION_TYPE_BUY
                    else mt5.symbol_info_tick(symbol).ask
                ),
                "deviation": 2000,
                "magic": tf_magic,
                "comment": "expire_by_reset",
            }

            result = mt5.order_send(close_request)

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(
                    f"Closed expired POSITION on {symbol} | "
                    f"opened={open_time} | reset={cutoff}"
                )
            else:
                logging.info(f"Failed to close POSITION on {symbol}: {result.retcode}")


def sort_symbols(symbols_list, htf):
    premium_symbols = []
    discount_symbols = []
    
    for symbol in symbols_list:
        
        result = retry(get_, symbol, htf)
        if not result:
            continue
        df, symbol_info, _, _ = result
        
        highs, lows = find_swings(df, major=True, atr_multiplier=2.0, atr_period=7)

        if highs and lows:
            last_high, last_low = highs[-1][1], lows[-1][1]

            if last_high and last_low:
                premium_level = last_low + (last_high - last_low) * 0.75
                discount_level = last_low + (last_high - last_low) * 0.25

                if symbol_info.bid >= premium_level and symbol_info.bid <= last_high:
                    premium_symbols.append(symbol)
                elif symbol_info.bid <= discount_level and symbol_info.bid >= last_low:
                    discount_symbols.append(symbol)
                else:
                    continue
            else:
                continue
        else:
            continue

    return premium_symbols, discount_symbols

def run_model_for_symbols(model_func, symbols, timeframe, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=len(symbols)) as executor:
        futures = {executor.submit(model_func, symbol, timeframe, *args, **kwargs): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                future.result()
            except MaxOrdersReached:
                logging.warning("Max orders reached — stopping further submissions for this cycle")
                break
            except Exception as e:
                logging.error(f"Error in {model_func.__name__} for {symbol}: {e}")
                
    
def chuks_type1_model(symbol, timeframe, symbols_cache, function_dict, symbol_status, order_params, tag):
    if symbol_status[symbol]["status"] == "disabled":
        logging.info(f"Trading on {symbol} on {timeframe} is currently disabled till {symbol_status[symbol]["reset_date"]}")
        return
    
    magic_tf = 100000 if timeframe == mt5.TIMEFRAME_M5 else 200000

    if symbol_status[symbol]["status"] == "default":
        positions = mt5.positions_get(symbol=symbol) or []
        orders = mt5.orders_get(symbol=symbol) or []

        # Filter by magic number for THIS timeframe
        tf_positions = [
            p for p in positions
            if getattr(p, "magic", None) == magic_tf
        ]

        tf_orders = [
            o for o in orders
            if getattr(o, "magic", None) == magic_tf
        ]

        if tf_positions or tf_orders:
            logging.info(
                f"{symbol} skipped ({timeframe}): "
                f"active position or pending order exists for this timeframe (magic={magic_tf})"
            )
            return
        
    result = retry(get_, symbol, timeframe)    
    if not result:
        return
    df, symbol_info, spread, balance = result

    tick_size = symbol_info.trade_tick_size 
    rd = min(abs(Decimal(str(tick_size)).as_tuple().exponent), symbol_info.digits)

    highs, lows = find_swings(df, dp=rd)
    all_swings = highs + lows
    all_swings.sort(key=lambda x: x[2])
    indexes = [swing[2] for swing in all_swings]
    index_map = {index: i for i, index in enumerate(indexes)}

    risk_reward = order_params[symbol][str(timeframe)]["risk_reward"]
    atr_multiplier = order_params[symbol][str(timeframe)]["atr_multiplier"]
    sensitivity = order_params[symbol][str(timeframe)]["sensitivity"]
    modified_stop = order_params[symbol][str(timeframe)]["modified_stop"]

    # Set buy/sell parameters
    if tag == "1B":
        pattern = ["L", "H", "L", "H", "L", "H"]
        order_type = mt5.ORDER_TYPE_BUY_LIMIT
        entry_price_func = lambda df, idxs, prices: min(df.iloc[idxs[0]-1 : idxs[0]+2]["high"].min(), prices[4])
        sl_func = lambda prices, atr, spread, atr_multiplier: prices[2] - (atr_multiplier * atr) + spread
        tp_func = lambda entry, sl_ticks, spread, prices, risk_reward: max(entry + (risk_reward * sl_ticks) + spread, prices[5] + spread)
    else:
        pattern = ["H", "L", "H", "L", "H", "L"]
        order_type = mt5.ORDER_TYPE_SELL_LIMIT
        entry_price_func = lambda df, idxs, prices: max(df.iloc[idxs[0]-1 : idxs[0]+2]["low"].max(), prices[4])
        sl_func = lambda prices, atr, spread, atr_multiplier: prices[2] + (atr_multiplier * atr) - spread
        tp_func = lambda entry, sl_ticks, spread, prices, risk_reward: min(entry - (risk_reward * sl_ticks) - spread, prices[5] - spread)

    for pattern_match in search_patterns(all_swings, df, pattern, indexes, index_map, tag, function_dict, sensitivity):
        prices = tuple([p[0] for p in pattern_match])

        if prices not in symbols_cache[symbol]:
            logging.info(f"Present pattern on {symbol}: {prices}")

            indices = [p[1] for p in pattern_match]
            entry_price = entry_price_func(df, indices, prices)
            atr = compute_atr(df).iloc[-1].round(rd)

            starting_balance = 100.0
            multiplier = balance // starting_balance if balance > starting_balance else 1

            risk = round((starting_balance * multiplier) * 0.1, 1)
            
            sl = round(sl_func(prices, atr, spread, atr_multiplier), rd)

            sl_ticks = round(abs(entry_price - sl), rd)
            if modified_stop:
                entry_price = prices[2]
                sl =  round(entry_price - sl_ticks + spread if tag == "1B" else entry_price + sl_ticks - spread, rd)

            n_ticks = round(sl_ticks / tick_size, 1)
            tick_value = symbol_info.trade_tick_value

            lot_precision = abs(Decimal(str(symbol_info.volume_step)).as_tuple().exponent)
            lot = round(risk / (tick_value * n_ticks), lot_precision)
            lot = max(symbol_info.volume_min, min(lot, symbol_info.volume_max))
            actual_risk = round(lot * tick_value * n_ticks, 1)

            if actual_risk < risk:
                sl_ticks = (risk * tick_size) / (lot * tick_value)
                sl = round(entry_price - sl_ticks if tag == "1B" else entry_price + sl_ticks, rd)
            elif actual_risk > risk and actual_risk - risk > 0.1:
                logging.info(f"Risk too high for this trade: {actual_risk}")
                continue

            tp = round(tp_func(entry_price, sl_ticks, spread, prices, risk_reward), rd)            

            if place_order(order_type, sl, tp, entry_price, lot, symbol, timeframe):
                symbols_cache[symbol].add(prices)

                try:
                    visualize_entry(df, pattern_match, symbol, timeframe, order_type, entry_price)
                except Exception as e:
                    logging.error(f"Error while plotting entry: {e}")
            
                if symbol_status[symbol]["status"] == "default":
                    break


def trading_job():
    """Main trading function"""
    if not retry(initialize_mt5):
        return

    
    login = 5925197
    password = "Ttobs_der1v"
    server = "Deriv-Demo"

    if not retry(login_mt5, login, password, server):
        return
    
    if mt5.orders_total() + mt5.positions_total() >= EXPOSURE_LIMIT:
        logging.info("Maximum exposure limit reached. No new trades will be placed.")
        mt5.shutdown()
        return
    
    M5_symbol_status, M5_clear = load_M5_symbol_status()
    M15_symbol_status, M15_clear = load_M15_symbol_status()

    if M5_clear:
        clear_symbols_cache("M5_trades.json")

    if M15_clear:
        clear_symbols_cache("M15_trades.json")
    
    create_symbols_cache(M5_trades_file)
    create_symbols_cache(M15_trades_file)
    
    M5_symbols_cache = load_symbols_cache(M5_trades_file)
    M15_symbols_cache = load_symbols_cache(M15_trades_file)
    
    M5_buys, M5_sells = sort_symbols(M5_htf_symbols, mt5.TIMEFRAME_H1)
    M15_buys, M15_sells = sort_symbols(M15_htf_symbols, mt5.TIMEFRAME_H4)

    M5_buys = M5_buys + M5_symbols
    M5_sells = M5_sells + M5_symbols

    M15_buys = M15_buys + M15_symbols
    M15_sells = M15_sells + M15_symbols

    model_functions_dict = {
        "1B":[type1_buy_check, type1_buy_prune_function],
        "1S":[type1_sell_check, type1_sell_prune_function]
    }

    close_expired_orders_and_trades(M5_symbol_status, tf_magic=100000)
    close_expired_orders_and_trades(M15_symbol_status, tf_magic=200000)

    order_params = load_order_params()

    try:
        if M5_buys:
            logging.info(f"Looking for type1 buy setups on 5M")
            run_model_for_symbols(chuks_type1_model, M5_buys, mt5.TIMEFRAME_M5, M5_symbols_cache, model_functions_dict, M5_symbol_status, order_params, tag="1B")

        if M5_sells:
            logging.info(f"Looking for type1 sell setups on 5M")
            run_model_for_symbols(chuks_type1_model, M5_sells, mt5.TIMEFRAME_M5, M5_symbols_cache, model_functions_dict, M5_symbol_status, order_params, tag="1S")

        logging.info(f"Looking for type1 buy setups on 15M")
        run_model_for_symbols(chuks_type1_model, M15_buys, mt5.TIMEFRAME_M15, M15_symbols_cache, model_functions_dict, M15_symbol_status, order_params, tag="1B")
        logging.info(f"Looking for type1 sell setups on 15M")
        run_model_for_symbols(chuks_type1_model, M15_sells, mt5.TIMEFRAME_M15, M15_symbols_cache, model_functions_dict, M15_symbol_status, order_params, tag="1S")
    finally:
        save_symbols_cache(M5_symbols_cache, M5_trades_file)
        save_symbols_cache(M15_symbols_cache, M15_trades_file)
        mt5.shutdown()
        logging.info("MT5 connection closed")
        logging.info("Trading bot stopped")


if __name__ == "__main__":

    scheduler = BlockingScheduler()
    scheduler.add_job(
        trading_job, 
        'cron', 
        day_of_week='mon-sun', 
        #hour='1-19/2',
        hour="*",
        minute="*/5",
        #minute="15,30,45,0",
        timezone='Africa/Lagos'
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass