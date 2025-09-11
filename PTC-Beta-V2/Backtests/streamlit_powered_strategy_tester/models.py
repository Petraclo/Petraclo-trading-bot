# models.py
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Boolean, DateTime, Text
)
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()
engine = create_engine("sqlite:///backtests.db")  # local file
SessionLocal = sessionmaker(bind=engine)

class Run(Base):
    __tablename__ = "runs"
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Parameters
    symbol = Column(String)
    timeframe = Column(String)
    begin_date = Column(String)
    end_date = Column(String)
    initial_balance = Column(Float)
    r_r = Column(Integer)
    atr_multiplier = Column(Float)
    sensitivity = Column(Integer)
    htf_analysis = Column(Boolean)
    modified_stop = Column(Boolean)
    adaptive = Column(Boolean)

    # Results
    pnl_usd = Column(Float)
    return_pct = Column(Float)
    win_rate_pct = Column(Float)
    max_dd_pct = Column(Float)
    avg_weekly_pnl_usd = Column(Float)
    avg_trades = Column(Float)
    small_acct_score = Column(Float)

    equity_curve_json = Column(Text)  # optional (for plotting later)

    __table_args__ = (
        UniqueConstraint(
            "symbol", "timeframe", "begin_date", "end_date", "initial_balance",
            "r_r", "atr_multiplier", "sensitivity", "htf_analysis",
            "modified_stop", "adaptive",
            name="uq_run_params"
        ),
    )

def init_db():
    Base.metadata.create_all(engine)
