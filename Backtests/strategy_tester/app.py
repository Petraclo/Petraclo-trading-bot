import streamlit as st
import pandas as pd
from datetime import datetime, date, time, timedelta, timezone
import math
from models import SessionLocal, Run, init_db
from sqlalchemy.exc import IntegrityError
import json
from rag_retrieval import retrieve_runs, build_context
from groq_client import ask_groq
from rag_store import upsert_run_to_index

# Import your backend (will run its top-level initialization but uses the functions)
import strat_tester as tester  # file you uploaded. :contentReference[oaicite:1]{index=1}

# --- UI ---
st.set_page_config(page_title="Strategy Tester UI", layout="wide")
st.title("Strategy Tester")

tabs = st.tabs(["Backtest", "AI Consultant"])

# Helper: timeframe options (based on your script)
TIMEFRAME_OPTIONS = ["5M", "15M", "1H", "4H"]
TF_TO_MT5 = {
    "5M": tester.mt5.TIMEFRAME_M5,
    "15M": tester.mt5.TIMEFRAME_M15,
    "1H": tester.mt5.TIMEFRAME_H1,
    "4H": tester.mt5.TIMEFRAME_H4,
}

init_db()

# Symbol list: filter available symbols to volatility indices
def available_volatility_symbols():
    try:
        syms = tester.mt5.symbols_get() or []
        vol_syms = sorted({s.name for s in syms if "volatility" in s.name.lower()})
        # Fallback to a common list if none found
        if not vol_syms:
            vol_syms = [
                "Volatility 10 Index",
                "Volatility 25 Index",
                "Volatility 50 Index",
                "Volatility 75 Index",
                "Volatility 100 Index",
            ]
        return vol_syms
    except Exception as e:
        # If MT5 not available, provide a fallback list
        return [
            "Volatility 10 Index",
            "Volatility 25 Index",
            "Volatility 50 Index",
            "Volatility 75 Index",
            "Volatility 100 Index",
        ]

from sqlalchemy.exc import IntegrityError

def save_run(symbol, timeframe, begin_dt, end_dt, initial_balance, r_r,
             atr_multiplier, sensitivity, htf_analysis, modified_stop, adaptive,
             summary_row, equity_path):
    session = SessionLocal()
    try:
        new_run = Run(
            symbol=symbol,
            timeframe=timeframe,
            begin_date=str(begin_dt.date()),
            end_date=str(end_dt.date()),
            initial_balance=initial_balance,
            r_r=r_r,
            atr_multiplier=atr_multiplier,
            sensitivity=sensitivity,
            htf_analysis=htf_analysis,
            modified_stop=modified_stop,
            adaptive=adaptive,
            pnl_usd=summary_row["PnL (USD)"],
            return_pct=summary_row["Return (%)"],
            win_rate_pct=summary_row["Win_Rate (%)"],
            max_dd_pct=summary_row["Max_Drawdown (%)"],
            avg_weekly_pnl_usd=summary_row["Average_Weekly_PnL (USD)"],
            avg_trades=summary_row["Avg_Trade_Freq"],   # or replace with total trades
            equity_curve_json=json.dumps(equity_path),
        )

        session.add(new_run)
        session.commit()
        session.refresh(new_run)  # ✅ safe, still persistent here

        upsert_run_to_index(new_run)

    except IntegrityError:
        session.rollback()
        # fetch and update existing row
        existing = session.query(Run).filter_by(
            symbol=symbol,
            timeframe=timeframe,
            begin_date=str(begin_dt.date()),
            end_date=str(end_dt.date()),
            initial_balance=initial_balance,
            r_r=r_r,
            atr_multiplier=atr_multiplier,
            sensitivity=sensitivity,
            htf_analysis=htf_analysis,
            modified_stop=modified_stop,
            adaptive=adaptive
        ).first()

        if existing:
            session.commit()
            session.refresh(existing)

            upsert_run_to_index(existing)

    finally:
        session.close()

with tabs[0]:
    with st.sidebar:
        st.header("Backtest parameters")
        symbol = st.selectbox("Symbol (Volatility indices)", available_volatility_symbols())
        timeframe = st.selectbox("Timeframe", TIMEFRAME_OPTIONS)
    
        # Dates: begin and end. enforce end <= 2025-08-18
        st.write("Date range (end must be on or before 2025-09-08)")
        begin_date = st.date_input("Begin date", value=date(2024, 12, 26), key="begin")
        max_end = date(2025, 9, 8)
        end_date = st.date_input("End date", value=min(date(2025, 9, 8), date.today()), max_value=max_end, key="end")
    
        # Ensure begin < end on run
        initial_balance = st.number_input("Initial balance (USD)", min_value=1.0, value=200.0, step=10.0, format="%.2f")
        
        with open("capital.json", "w") as f:
            json.dump({"capital": initial_balance}, f, indent=4)

        r_r = st.number_input("R : R (integer >=1)", min_value=1, value=3, step=1)
        atr_multiplier = st.number_input("ATR multiplier (float >=1)", min_value=1.0, value=1.5, step=0.1, format="%.2f")
        sensitivity = st.number_input("Sensitivity (integer >=1)", min_value=1, value=2, step=1)
        htf_analysis = st.checkbox("Higher timeframe analysis (HTF)", value=False)
        modified_stop = st.checkbox("Modified stop", value=False)
        adaptive = st.checkbox("Adaptive", value=False)
    
        run_btn = st.button("Run Backtest")
    
    
    # --- Backtest runner (uses functions from StrategyTester.py) ---
    def dt_from_date(d: date) -> datetime:
        # use midnight UTC to match your script's timezone-aware datetimes
        return datetime(d.year, d.month, d.day, 0, 0, tzinfo=timezone.utc)
    
    def run_full_backtest(symbol, timeframe_label, begin_dt, end_dt, initial_balance,
                          r_r, atr_multiplier, sensitivity, htf_analysis, modified_stop, adaptive):
        timeframe_mt5 = TF_TO_MT5[timeframe_label]
        # mapping string name for get_end_date
        tf_for_get_end = timeframe_label
    
        balance = float(initial_balance)
        overall_equity_values = [balance]
        weekly_md_list = []
        weekly_pnls = []
        no_of_trades_per_week = []
        profitable_trades = 0
    
        start_date = begin_dt
        end_marker = end_dt
    
        progress = st.progress(0)
        steps = 100
        iters = 0
        total_iters_guess = 70  # used to increment progress bar sensibly
    
        # loop using your get_end_date to form chunks (same logic as your main)
        while start_date < end_marker:
            end_date_chunk = tester.get_end_date(tf_for_get_end, start_date)
            # clamp
            if end_date_chunk is None:
                # fallback: advance by 1 day
                end_date_chunk = start_date + timedelta(days=1)
            end_date_chunk = dt_from_date(end_date_chunk)
            if end_date_chunk > end_marker:
                end_date_chunk = end_marker
    
            # fetch rates for chunk
            df = tester.fetch_rates(symbol, timeframe_mt5, start_date, end_date_chunk)
            if len(df) == 0:
                # advance to next window to avoid infinite loops
                start_date = end_date_chunk
                continue
    
            # call your backtester for the chunk
            try:
                equity_end_of_week, equity_values_week, md_week_pct, profitable_trades_for_week, n_signals = tester.backtester(
                    df,
                    balance,
                    symbol,
                    risk_reward_ratio=int(r_r),
                    atr_multiplier=float(atr_multiplier),
                    sensitivity=int(sensitivity),
                    htf_analysis=bool(htf_analysis),
                    modified_stop=bool(modified_stop),
                    adaptive=bool(adaptive)
                )
            except Exception as e:
                st.error(f"Error running backtester on chunk starting {start_date.date()}: {e}")
                break
    
            # update running balance and metrics
            balance = equity_end_of_week
            if balance <= 0:
                st.warning("Account blown during backtest.")
                break
    
            no_of_trades_per_week.append(len(equity_values_week) - 1)
            weekly_md_list.append(md_week_pct)
            weekly_pnls.append(round(equity_end_of_week - overall_equity_values[-1], 2))
            
            profitable_trades += profitable_trades_for_week
    
            if len(equity_values_week) > 1:
                overall_equity_values.extend(equity_values_week[1:])
    
            # advance window
            start_date = end_date_chunk
    
            # progress update
            iters += 1
            pct = min(1.0, iters / max(1, total_iters_guess))
            progress.progress(int(pct * 100))
    
        # final metrics
        overall_md_pct, overall_peak = tester.calculate_max_drawdown(overall_equity_values)
        total_pnl = round(balance - initial_balance, 2)
        total_return_pct = round((balance / initial_balance - 1) * 100, 2) if initial_balance else 0.0
        max_weekly_md_pct = round(max(weekly_md_list)*100, 2) if weekly_md_list else 0.0
        avg_weekly_pnl = round(pd.Series(weekly_pnls).mean(), 2) if weekly_pnls else 0.0
        total_trades = int(sum(no_of_trades_per_week)) if no_of_trades_per_week else 0
        win_rate = round((profitable_trades/total_trades) * 100, 2) if total_trades else 0.0
        avg_trades_per_week = round(pd.Series(no_of_trades_per_week).mean(), 1) if no_of_trades_per_week else 0.0
    
        summary = {
            "Start_Date": begin_dt,
            "End_Date": end_dt,
            "Symbol": symbol,
            "Timeframe": timeframe_label,
            "R:R": r_r,
            "ATR_Multiplier": atr_multiplier,
            "Algo_Sensitivity": sensitivity,
            "Adaptive": adaptive,
            "HTF_Analysis": htf_analysis,
            "Modified_Stop": modified_stop,
            "PnL (USD)": total_pnl,
            "Return (%)": total_return_pct,
            "Win_Rate (%)": win_rate,
            "Avg_Trade_Freq": avg_trades_per_week,
            "Max_Drawdown (%)": round(overall_md_pct * 100, 2),
            "Peak": round(overall_peak, 2),
            "Lowest point": round(min(overall_equity_values), 2) if overall_equity_values else None,
            "Max_Weekly_Drawdown (%)": max_weekly_md_pct,
            "Average_Weekly_PnL (USD)": avg_weekly_pnl
        }
    
        return summary, overall_equity_values, total_trades, total_pnl, win_rate
    
    # --- Run/backtest button action ---
    if run_btn:
        # basic validation
        if begin_date >= end_date:
            st.error("Begin date must be earlier than end date.")
        elif end_date > date(2025, 9, 8):
            st.error("End date must be on or before 2025-09-08.")
        else:
            # convert to datetimes matching your script's timezone usage
            st.info("Starting backtest — this may take a while depending on symbol/timeframe and MT5 connection.")
            with st.spinner("Running backtest..."):
                begin_dt = dt_from_date(begin_date)
                end_dt = dt_from_date(end_date)
                summary_row, equity_path, total_trades, total_pnl, win_rate = run_full_backtest(
                    symbol, timeframe, begin_dt, end_dt, initial_balance,
                    r_r, atr_multiplier, sensitivity, htf_analysis, modified_stop, adaptive
                )
    
            # display summary
            st.subheader("Backtest summary")
            summary_df = pd.DataFrame([summary_row])
            st.table(summary_df.T.rename(columns={0: "Value"}))
    
            # equity curve
            if equity_path:
                st.subheader("Equity curve")
                eq_series = pd.Series(equity_path)
                st.line_chart(eq_series, color='rgba(0, 138, 0, 0.8)')
    
            st.markdown(f"**Total trades:** {total_trades}  •  **Total PnL (USD):** {total_pnl}  •  **Win rate:** {win_rate}%")
    
        save_run(symbol, timeframe, begin_dt, end_dt, initial_balance, r_r, atr_multiplier, sensitivity, htf_analysis, modified_stop, adaptive, summary_row, equity_path)
        
        st.subheader("Past Backtest Runs")
        session = SessionLocal()
        runs = session.query(Run).order_by(Run.created_at.desc()).limit(10).all()
        session.close()
        
        if runs:
            hist_df = pd.DataFrame([{
                "id": r.id,
                "created_at": r.created_at,
                "symbol": r.symbol,
                "timeframe": r.timeframe,
                "pnl": r.pnl_usd,
                "return%": r.return_pct,
                "win_rate%": r.win_rate_pct,
                "max_dd%": r.max_dd_pct,
            } for r in runs])
            st.dataframe(hist_df)

with tabs[1]:
    st.header("AI Consultant")

    c1, c2 = st.columns(2)
    f_symbol = c1.text_input("Symbol filter (optional)")
    f_tf = c2.text_input("Timeframe filter (optional)")
    q = st.text_area("Ask the advisor", placeholder="e.g. Which V75 15M runs had lowest drawdowns?")

    if st.button("Ask Advisor"):
        rows, docs = retrieve_runs(q, symbol=f_symbol or None, timeframe=f_tf or None, top_k=5)
        if not rows:
            st.warning("No matching runs found.")
        else:
            ctx_text = build_context(rows, docs)
            answer = ask_groq(q, ctx_text)
            st.markdown("### Advisor’s Recommendation")
            st.write(answer)

            st.subheader("Evidence (Top Runs)")
            st.dataframe([{
                "ID": r.id,
                "Symbol": r.symbol,
                "TF": r.timeframe,
                "Win%": r.win_rate_pct,
                "MaxDD%": r.max_dd_pct,
                "PnL": r.pnl_usd,
                "Return%": r.return_pct,
                "Trades": r.avg_trades,
                "Date": r.created_at
            } for r in rows])