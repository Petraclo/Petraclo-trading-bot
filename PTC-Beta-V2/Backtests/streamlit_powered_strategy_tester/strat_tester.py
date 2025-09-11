import pandas as pd
import numpy as np
import random
from config import TradeParameters
import MetaTrader5 as mt5
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta

params = TradeParameters()

mt5.initialize()#"C:\\Program Files\\MetaTrader 5 Terminal - 4\\terminal64.exe")
mt5.login(params.login, params.password, params.server)


## Helper Functions
def read_value():
    with open("capital.json", "r") as f:
        import json
        data = json.load(f)
        starting_balance = data.get("capital", 100)
    return starting_balance

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
    if df["low"].iloc[idxs[2]+4 : idxs[5]+4].min() <= prices[2]:
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
    if df["high"].iloc[idxs[2]+4 : idxs[5]+4].max() >= prices[2]:
        return False
    
    if prices[4] < (prices[1] + prices[2]) / 2:
        return False
    return (
        prices[3] < prices[1] and
        prices[2] > prices[0] and
        prices[2] > prices[4] and
        prices[5] < prices[3]
    )

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

def higher_timeframe_analysis(end_time, tf, symbol, entry_price):
    htf_map = {
        "5M": [mt5.TIMEFRAME_H1, end_time-timedelta(days=10)],
        "15M": [mt5.TIMEFRAME_H4, end_time-timedelta(days=40)],
        "1H": [mt5.TIMEFRAME_D1, end_time-timedelta(days=150)],
        "4H": [mt5.TIMEFRAME_W1, end_time-timedelta(days=1000)]
              }

    df = pd.DataFrame(mt5.copy_rates_range(symbol, htf_map[tf][0], htf_map[tf][1], end_time))
    
    highs, lows = find_swings(df, major=True, atr_multiplier=2.0, atr_period=7)

    last_high, last_low = highs[-1][1], lows[-1][1]

    premium_level = last_low + (last_high - last_low) * 0.75
    discount_level = last_low + (last_high - last_low) * 0.25
        
    if entry_price >= premium_level and entry_price <= last_high:
        return "sell"
    elif entry_price <= discount_level and entry_price >= last_low:
        return "buy"
    else:
        return "no signal"


# ## Main function

# In[ ]:


def search_patterns(events, df, pattern, sensitivity, idxs, idx_map, tag_dict, tag, idx_start=0, prev_idx=None, chosen=None):
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
            yield from search_patterns(events, df, pattern, sensitivity, idxs, idx_map, tag_dict, tag, j+1, current_idx, chosen)
            chosen.pop()

        else:
            # Subsequent matches: check adjacency constraint
            if i in idxs[prev_idx + 1:prev_idx + (1+sensitivity)]:
                if tag_dict[tag][1](pos, p, chosen):
                    continue
                current_idx = idx_map.get(i)
                chosen.append((p,i))
                yield from search_patterns(events, df, pattern, sensitivity, idxs, idx_map, tag_dict, tag, j+1, current_idx, chosen)
                chosen.pop()
                
            else:
                return

def chuks_type1_model(df, symbol, sensitivity, htf_analysis):
    signals = np.zeros((len(df), 4), dtype=int)

    symbol_info = mt5.symbol_info(symbol)

    tick_size = symbol_info.trade_tick_size 
    rd = min(abs(Decimal(str(tick_size)).as_tuple().exponent), symbol_info.digits)

    highs, lows = find_swings(df, dp=rd)
    all_swings = highs + lows
    all_swings.sort(key=lambda x: x[2])
    indexes = [swing[2] for swing in all_swings]
    index_map = {index: i for i, index in enumerate(indexes)}

    buy_pattern = ["L", "H", "L", "H", "L", "H"]
    sell_pattern = ["H", "L", "H", "L", "H", "L"]

    function_dict = {
        "1B":[type1_buy_check, type1_buy_prune_function],
        "1S":[type1_sell_check, type1_sell_prune_function]
    }
    
    for buy_pattern_match in search_patterns(all_swings, df, buy_pattern, sensitivity, indexes, index_map, function_dict, tag="1B"):
        
        prices = [p[0] for p in buy_pattern_match]
        indices = [p[1] for p in buy_pattern_match]

        slice_ = df.iloc[indices[0]-1 : indices[0]+2]

        min_high = slice_["high"].min()
        entry_price = min(min_high, prices[4])
        index = indices[-1] + 4

        if htf_analysis:
            time_of_entry = df.iloc[index]["time"]
            tf = infer_timeframe(df)
            if tf:
                if higher_timeframe_analysis(time_of_entry, tf, symbol, entry_price) != "buy":
                    continue
                
        
        df_after_pattern = df.iloc[index:]
        df2 = df_after_pattern[df_after_pattern["low"] <= entry_price]
        if not df2.empty:
            entry = df2.index[0]
            signals[entry][0] = 1
            signals[entry][1] = prices[2]
            signals[entry][2] = entry_price
            signals[entry][3] = prices[5]

    for sell_pattern_match in search_patterns(all_swings, df, sell_pattern, sensitivity, indexes, index_map, function_dict, tag="1S"):
        
        prices = [p[0] for p in sell_pattern_match]
        indices = [p[1] for p in sell_pattern_match]

        slice_ = df.iloc[indices[0]-1 : indices[0]+2]

        min_low = slice_["low"].min()
        entry_price = min(min_low, prices[4])
        index = indices[-1] + 4

        if htf_analysis:
            time_of_entry = df.iloc[index]["time"]
            tf = infer_timeframe(df)
            if tf:
                if higher_timeframe_analysis(time_of_entry, tf, symbol, entry_price) != "sell":
                    continue
                
        
        df_after_pattern = df.iloc[index:]
        df2 = df_after_pattern[df_after_pattern["high"] >= entry_price]
        if not df2.empty:
            entry = df2.index[0]
            signals[entry][0] = -1
            signals[entry][1] = prices[2]
            signals[entry][2] = entry_price
            signals[entry][3] = prices[5]

    return signals


def sim_trade(df, capital, entry_point, symbol, risk_reward_ratio, atr_multiplier, modified_stop):
    """Simulates a trade with a given capital.

    Args:
        df: price dataframe
        capital: money used to make the trade
        entry_dict: dict containing strategy entry info

    Returns:
        new_capital: new value of capital after trade
    """
    new_capital = capital
    entry_dict = entry_point

    result = backtest_O1C_tp_sl_function(
        df, entry_dict["pos"], entry_dict['signal'],
        symbol, new_capital, risk_reward_ratio,
        atr_multiplier, modified_stop
    )
    if not result or any(x is None for x in result):
        # skip this trade, capital unchanged
        return new_capital, False

    tp, sl, tp_ticks, sl_ticks, lot, risk = result
    entry_dict.update({
        "take_profit": tp,
        "stop_loss": sl,
        "tp_pips": tp_ticks,
        "sl_pips": sl_ticks,
        "lot": lot,
        "risk": risk
    })
    
    lot = entry_dict["lot"]
    #spread_pips = random.uniform(0.2, 2.5)
    slippage_pips = random.uniform(0.0, 1.0)

    df_after_entry = df[df["time"] > entry_dict["date"]]
    df2 = df_after_entry.copy()

    profit = None
    if entry_dict["type"] == "long":
    
        tp_hit = df2[df2["high"] >= entry_dict["take_profit"]]
        sl_hit = df2[df2["low"] <= entry_dict["stop_loss"]]
    
        if not tp_hit.empty and not sl_hit.empty:
            profit = tp_hit.index[0] < sl_hit.index[0]
        elif not tp_hit.empty:
            profit = True
        elif not sl_hit.empty:
            profit = False

    elif entry_dict["type"] == "short":

        tp_hit = df2[df2["low"] <= entry_dict["take_profit"]]
        sl_hit = df2[df2["high"] >= entry_dict["stop_loss"]]

        if not tp_hit.empty and not sl_hit.empty:
            profit = tp_hit.index[0] < sl_hit.index[0]
        elif not tp_hit.empty:
            profit = True
        elif not sl_hit.empty:
            profit = False

    if profit is True:
        new_capital += entry_dict["risk"] * (round(entry_dict["tp_pips"]/entry_dict["sl_pips"], 0))
    elif profit is None:
        new_capital = new_capital
    else:
        new_capital -= entry_dict["risk"]

    return new_capital, True


def get_end_date(timeframe, start_date: datetime):
    if timeframe == "5M":
        # 3 days from start_date
        return start_date + timedelta(days=3)

    if timeframe == "15M":
        # Next Monday
        days_ahead = (7 - start_date.weekday()) % 7  # 0=Monday ... 6=Sunday
        if days_ahead == 0:  # already Monday → move to next Monday
            days_ahead = 7
        return start_date + timedelta(days=days_ahead)

    if timeframe == "1H":
        # First day of next month
        return (start_date + relativedelta(months=1)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    if timeframe == "4H":
        # First day of next quarter
        month = ((start_date.month - 1) // 3 + 1) * 3 + 1
        year = start_date.year
        if month > 12:
            month = 1
            year += 1
        return datetime(year, month, 1)

    if timeframe == "1D":
        # First day of next year
        return datetime(start_date.year + 1, 1, 1)

    # Unknown timeframe
    return None

def infer_timeframe(df, datetime_col="time"):
    diffs = df[datetime_col].diff().dropna()
    if diffs.empty:
        return None
    
    # Most common interval (mode)
    interval = diffs.mode()[0]
    minutes = interval.total_seconds() / 60
    
    # Map to common timeframes
    mapping = {
        5: "5M",
        15: "15M",
        60: "1H",
        240: "4H",
    }
    
    return mapping.get(int(minutes))

def calculate_max_drawdown(equity_values):
    """
    Returns:
        max_dd_pct: maximum drawdown as fraction (0..1)
        peak_value: peak equity value observed
    """
    if not equity_values:
        return 0.0, 0.0
    peak = equity_values[0]
    peak_value = peak
    max_dd = 0.0
    for v in equity_values:
        if v > peak:
            peak = v
            peak_value = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd, peak_value



def backtest_O1C_tp_sl_function(df, i, signal, symbol, capital, risk_reward_ratio, atr_multiplier, modified_stop):
    ### Calculating lot using captal at the start of the week instaed of updating along side equity ###
    si = mt5.symbol_info(symbol)
    
    tick_size = si.trade_tick_size
    spread = si.spread * tick_size 
    rd = min(abs(Decimal(str(si.trade_tick_size)).as_tuple().exponent), si.digits)
    
    atr = compute_atr(df.iloc[:i]).iloc[-1].round(rd)

    buy = signal[0] == 1 

    starting_balance = read_value()

    if capital > starting_balance:
        multiplier = capital // starting_balance

    else:
        multiplier = 1
        
    risk = round((starting_balance * multiplier) * 0.1, 1)
    
    sl = round(signal[1] - (atr_multiplier * atr) + spread if buy else signal[1] + (atr_multiplier * atr) - spread, rd)
    
    entry_price = signal[2]
    
    sl_ticks = round(abs(entry_price - sl), rd)
    if modified_stop:
        entry_price = signal[1]
        sl =  round(entry_price - sl_ticks + spread if buy else entry_price + sl_ticks - spread, rd)
        
    n_ticks = round(sl_ticks / tick_size, 1)
    tick_value = si.trade_tick_value

    lot_precision = abs(Decimal(str(si.volume_step)).as_tuple().exponent)
    lot = round(risk / (tick_value * n_ticks), lot_precision)
    lot = max(si.volume_min, min(lot, si.volume_max))
    actual_risk = round(lot * tick_value * n_ticks, 1)

    if actual_risk < risk and (risk - actual_risk) > 0.1:
        sl_ticks = (risk * tick_size) / (lot * tick_value)
        sl = round(entry_price - sl_ticks + spread if buy else entry_price + sl_ticks - spread, rd)
    elif actual_risk > risk and (actual_risk - risk) > 0.9:
        print(f"Risk too high: {actual_risk} vs intended {risk}")
        return None, None, None, None, None, None 

    tp_ticks = round(sl_ticks * risk_reward_ratio, rd)
    tp = round(max((entry_price + tp_ticks) + spread, signal[3] + spread) if buy else min((entry_price - tp_ticks) - spread, signal[3] - spread), rd)

    return tp, sl, tp_ticks, sl_ticks, lot, risk


def backtester_analyze_strategy(df, capital, symbol, strategy_function, tp_sl_function, **kwargs):
    
    analysis_df = df.copy()
    entry_signals = strategy_function(analysis_df, symbol, kwargs['sensitivity'], kwargs['htf_analysis'])
        
    entry_points = []
    for i, signal in enumerate(entry_signals):
        if signal[0] != 0 and i > 0:  

            close = analysis_df.iloc[i]['close']
            entry_points.append({
                'date': analysis_df.iloc[i]['time'],
                'price': close,
                'type': 'long' if signal[0] > 0 else 'short',
                'signal': signal,
                'pos': i
            })

    n_signals = len(entry_points)
                
    return entry_points, n_signals



def backtester(price_df, capital, symbol,
               verbose=True,
               strategy_function=chuks_type1_model,
               tp_sl_function=backtest_O1C_tp_sl_function,
               **kwargs):
    """
    Backtest for a single chunk (e.g. weekly chunk).
    Prints weekly PnL and weekly max drawdown, and returns:
        (final_equity, equity_values_list_for_this_chunk, weekly_md_pct)
    """
    df = price_df.copy()
    df["time"] = pd.to_datetime(df["time"], unit="s")
    
    entry_points, n_signals = backtester_analyze_strategy(
        df, capital, symbol,
        verbose=True,
        strategy_function=chuks_type1_model,
        tp_sl_function=backtest_O1C_tp_sl_function,
        **kwargs
    )

    equity = capital
    equity_values = [equity]
    profitable_trades = 0 

    for i, entry in enumerate(entry_points):
        equity, executed = sim_trade(df, equity, entry, symbol,
                                     kwargs['risk_reward_ratio'],
                                     kwargs['atr_multiplier'],
                                     kwargs['modified_stop'])
        if executed:
            equity_values.append(equity)
            if equity_values[-1] > equity_values[-2]:
                profitable_trades += 1
            else:
                if kwargs.get("adaptive") and i == 0:
                    break

    weekly_md_pct, weekly_peak = calculate_max_drawdown(equity_values)

    # Print weekly summary
    if verbose:
        print(
            f"=== Week {df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()} === \n"
            f"PnL: {round(equity - capital, 2)} USD ({round((equity/capital - 1)*100, 2)}%), "
            f"Weekly Max Drawdown: {round(weekly_md_pct*100, 2)}%, "
            f"Weekly Peak: {round(weekly_peak, 2)}, "
            f"Lowest Point: {round(min(equity_values), 2)} \n "
        )

    return round(equity, 2), equity_values, weekly_md_pct, profitable_trades, n_signals


def fetch_rates(symbol, timeframe, start, end):
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    df = pd.DataFrame(rates) if rates is not None else pd.DataFrame()
    try:
        df["time"] = pd.to_datetime(df["time"], unit="s")
    except Exception as e:
        print(e)
    return df

def init_mt5():
    mt5.initialize()
    mt5.login(login=params.login,password=params.password,server=params.server)
