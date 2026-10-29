"""
NSE Bulk & Block Deal Research Agent - STANDALONE SINGLE-FILE VERSION (Gemini Integrated)
This is a combined, standalone version of the nse_deal_agent package.
All relative imports have been resolved for single-file execution.
Anthropic/Claude has been replaced with Google Gemini (google.generativeai).
NO OTHER LOGIC OR FUNCTIONALITY HAS BEEN ALTERED.

Usage:
python nse_deal_agent_standalone.py --help
python nse_deal_agent_standalone.py --from 2026-09-28 --to 2026-10-03 --min-value 25 --top 3
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import logging
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import requests

# ============================================================================
# GLOBAL LOGGING & CONSTANTS
# ============================================================================
log = logging.getLogger("nse_deal_agent")

CRORE = 1e7
RISK_FREE = 0.065  # ~ Indian 91-day T-bill
DEAL_COLUMNS = ["date", "symbol", "name", "client", "side", "qty", "price", "deal_type"]
BASE = "https://www.nseindia.com"

# Comprehensive Browser-Grade Headers to bypass basic WAF/Cloudflare checks
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
    "Origin": "https://www.nseindia.com",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "X-Requested-With": "XMLHttpRequest",
    "Connection": "keep-alive",
}

# --- UPDATED FOR GEMINI ---
MODEL = "gemini-1.5-flash"
SYSTEM = """You are an equity research analyst covering Indian listed companies (NSE).
Analyze the bulk/block deal activity and technical context provided to judge the market sentiment.
Pay special attention to WHY large deals happened (promoter stake sale, PE/VC exit, FII/DII accumulation, QIP/OFS, block placement, pledge invocation, index rebalancing, etc.) and how the market interpreted them.
Do not invent facts. If you cannot find something in the provided context, state it clearly.
Return ONLY a valid JSON object (no markdown, no prose before or after it) of this exact shape:
{
  "sentiment": "BULLISH" | "MILDLY BULLISH" | "NEUTRAL" | "MILDLY BEARISH" | "BEARISH",
  "sentiment_score": <integer -100 to 100>,
  "deal_interpretation": "<who was on the other side of the deal and what it signals>",
  "key_drivers": ["<driver>", "..."],
  "upcoming_events": ["<results date, AGM, ex-date, lock-in expiry, etc.>", "..."],
  "risks": ["<risk>", "..."],
  "headlines": [{"date": "YYYY-MM-DD", "title": "...", "source": "...", "url": "..."}],
  "summary": "<3-5 sentence summary>"
}"""

DISCLAIMER = (
    "> **Disclaimer:** Generated automatically from exchange data, rule-based technicals and AI news "
    "research. It is NOT investment advice and the author is not a SEBI-registered adviser. F&O "
    "trading carries a high risk of loss (SEBI's study: ~9 in 10 individual F&O traders lose money). "
    "Verify every level, lot size and margin with your broker before trading."
)

# ============================================================================
# TELEGRAM & PDF INTEGRATION
# ============================================================================
def send_telegram_alert(message: str) -> bool:
    """Sends a Telegram text notification."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.warning("⚠️ Telegram credentials missing. Skipping text alert.")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        response = requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=10)
        response.raise_for_status()
        result = response.json()
        if result.get("ok"):
            log.info("✅ Telegram text alert sent.")
            return True
        log.error(f"❌ Telegram API error: {result.get('description')}")
        return False
    except requests.exceptions.RequestException as e:
        log.error(f"❌ Telegram network error: {e}")
        return False

def send_telegram_document(file_path: str, caption: str) -> bool:
    """Sends the generated report as a Telegram document."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.warning("⚠️ Telegram credentials missing. Skipping document send.")
        return False
    if not os.path.exists(file_path):
        log.error(f"❌ Report file not found at {file_path}.")
        return False
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    safe_caption = caption[:1024]
    try:
        with open(file_path, "rb") as f:
            files = {"document": (os.path.basename(file_path), f, "text/markdown")}
            data = {"chat_id": chat_id, "caption": safe_caption}
            response = requests.post(url, data=data, files=files, timeout=60)
            response.raise_for_status()
            result = response.json()
            if result.get("ok"):
                log.info(f"✅ Telegram document sent: {file_path}")
                return True
            log.error(f"❌ Telegram API error: {result.get('description')}")
            return False
    except Exception as e:
        log.error(f"❌ Error sending document: {e}")
        return False

def generate_pdf_report(md_path: str, pdf_path: str) -> bool:
    """Optional: Converts Markdown report to PDF using fpdf2."""
    try:
        from fpdf import FPDF
    except ImportError:
        log.warning("⚠️ 'fpdf2' not installed. Skipping PDF generation.")
        return False
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Helvetica", size=10)
        with open(md_path, "r", encoding="utf-8") as f:
            for line in f:
                safe_line = line.encode('latin-1', errors='replace').decode('latin-1')
                clean_line = safe_line.replace('#', '').replace('*', '').replace('`', '').strip()
                if clean_line:
                    pdf.multi_cell(0, 5, clean_line)
        pdf.output(pdf_path)
        log.info(f"✅ PDF report generated: {pdf_path}")
        return True
    except Exception as e:
        log.error(f"❌ PDF generation failed: {e}. Markdown report preserved.")
        return False

# ============================================================================
# 1. NSE CLIENT & UTILS (OPTIMIZED FOR CLOUD/WAF BYPASS)
# ============================================================================
def _nse_date(d: dt.date) -> str:
    return d.strftime("%d-%m-%Y")

def _parse_date(value: Any) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.date):
        return value
    s = str(value).strip()
    for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y", "%d-%B-%Y"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    try:
        return pd.to_datetime(s, dayfirst=True).date()
    except (ValueError, TypeError):
        return None

def _num(value: Any) -> float:
    if value is None or value == "" or value == "-":
        return float("nan")
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return float("nan")

def _pick(row: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default

def normalize_deal_rows(rows: list[dict], deal_type: str) -> pd.DataFrame:
    out = []
    for r in rows:
        out.append(
            {
                "date": _parse_date(_pick(r, "BD_DT_DATE", "date", "Date", "mTIMESTAMP")),
                "symbol": str(_pick(r, "BD_SYMBOL", "symbol", "Symbol", default="")).strip().upper(),
                "name": _pick(r, "BD_SCRIP_NAME", "name", "Security Name", default=""),
                "client": _pick(r, "BD_CLIENT_NAME", "clientName", "Client Name", default=""),
                "side": str(_pick(r, "BD_BUY_SELL", "buySell", "Buy/Sell", default="")).strip().upper(),
                "qty": _num(_pick(r, "BD_QTY_TRD", "qty", "Quantity Traded")),
                "price": _num(_pick(r, "BD_TP_WATP", "watp", "Trade Price / Wght. Avg. Price", "Price")),
                "deal_type": deal_type,
            }
        )
    df = pd.DataFrame(out, columns=DEAL_COLUMNS)
    if not df.empty:
        df = df[df["symbol"] != ""]
        df["side"] = df["side"].replace({"B": "BUY", "S": "SELL"})
    return df

def load_deals_csv(path: str, deal_type: str = "BULK") -> pd.DataFrame:
    with open(path, newline="", encoding="utf-8-sig") as f:
        text = f.read()
    reader = csv.DictReader(io.StringIO(text))
    rows = [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in reader]
    return normalize_deal_rows(rows, deal_type)

class NSEClient:
    def __init__(self, timeout: float = 15.0):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.timeout = timeout
        self._warmed = False

    def _warm(self) -> None:
        """Visits the NSE homepage to capture dynamic WAF/Cloudflare cookies."""
        if self._warmed:
            return
        
        try:
            self.session.cookies.clear()
            resp = self.session.get("https://www.nseindia.com/", timeout=self.timeout)
            resp.raise_for_status()
            time.sleep(random.uniform(1.5, 3.0))
            self._warmed = True
            log.info("✅ NSE session warmed up and cookies initialized successfully.")
        except requests.RequestException as e:
            log.warning(f"⚠️ NSE warm-up failed: {e}")
            self._warmed = False

    def get_json(self, path: str, params: dict | None = None, retries: int = 3) -> Any:
        """Fetches JSON with intelligent retry, exponential backoff, and automatic session refresh."""
        self._warm()
        url = f"{BASE}{path}"
        
        for attempt in range(retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                
                if resp.status_code in (401, 403):
                    log.warning(f"⚠️ Access denied ({resp.status_code}) for {path}. Refreshing session...")
                    self._warmed = False
                    self._warm()
                    continue 
                
                resp.raise_for_status()
                
                content_type = resp.headers.get('Content-Type', '')
                if 'application/json' not in content_type:
                    log.warning(f"⚠️ Expected JSON but got '{content_type}' for {path}. WAF may have blocked the request.")
                    log.debug(f"Response preview: {resp.text[:300]}")
                    self._warmed = False
                    continue
                    
                return resp.json()
                
            except requests.exceptions.JSONDecodeError:
                log.warning(f"⚠️ JSON decode error on attempt {attempt + 1}. Likely blocked by WAF.")
                self._warmed = False
            except requests.exceptions.RequestException as e:
                log.warning(f"⚠️ Request failed on attempt {attempt + 1}: {e}")
                
            if attempt < retries - 1:
                delay = (2 ** attempt) + random.uniform(1.0, 2.5)
                log.info(f"⏳ Retrying {path} in {delay:.2f} seconds...")
                time.sleep(delay)
                
        log.error(f"❌ Failed to fetch {path} after {retries} attempts.")
        return None

    def deals(self, start: dt.date, end: dt.date, kind: str = "bulk") -> pd.DataFrame:
        """Fetches historical bulk or block deals."""
        payload = self.get_json(
            f"/api/historical/{kind}-deals", 
            {"from": start.strftime("%d-%m-%Y"), "to": end.strftime("%d-%m-%Y")}
        )
        if not payload:
            return pd.DataFrame()
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        return normalize_deal_rows(rows, kind.upper())

    def latest_large_deals(self) -> pd.DataFrame:
        payload = self.get_json("/api/snapshot-capital-market-largedeal") or {}
        frames = [
            normalize_deal_rows(payload.get("BULK_DEALS_DATA") or [], "BULK"),
            normalize_deal_rows(payload.get("BLOCK_DEALS_DATA") or [], "BLOCK"),
        ]
        return pd.concat(frames, ignore_index=True)

    def price_history(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        df = self._nse_history(symbol, start, end)
        if df.empty:
            df = yfinance_history(symbol + ".NS", start, end)
        return df

    def _nse_history(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        frames = []
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + dt.timedelta(days=180), end)
            payload = self.get_json(
                "/api/historical/cm/equity",
                {"symbol": symbol, "series": '["EQ"]', "from": _nse_date(cursor), "to": _nse_date(chunk_end)},
            )
            rows = (payload or {}).get("data", []) if isinstance(payload, dict) else []
            if rows:
                frames.append(
                    pd.DataFrame(
                        {
                            "date": [_parse_date(_pick(r, "CH_TIMESTAMP", "mTIMESTAMP")) for r in rows],
                            "open": [_num(r.get("CH_OPENING_PRICE")) for r in rows],
                            "high": [_num(r.get("CH_TRADE_HIGH_PRICE")) for r in rows],
                            "low": [_num(r.get("CH_TRADE_LOW_PRICE")) for r in rows],
                            "close": [_num(r.get("CH_CLOSING_PRICE")) for r in rows],
                            "volume": [_num(r.get("CH_TOT_TRADED_QTY")) for r in rows],
                            "delivery_pct": [_num(r.get("COP_DELIV_PERC")) for r in rows],
                        }
                    )
                )
            cursor = chunk_end + dt.timedelta(days=1)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames).dropna(subset=["date", "close"])
        return df.drop_duplicates("date").sort_values("date").set_index("date")

    def index_history(self, start: dt.date, end: dt.date, index: str = "NIFTY 50") -> pd.DataFrame:
        payload = self.get_json("/api/historical/indicesHistory", {"indexType": index, "from": _nse_date(start), "to": _nse_date(end)})
        rows = []
        if isinstance(payload, dict):
            data = payload.get("data", {})
            rows = data.get("indexCloseOnlineRecords", []) if isinstance(data, dict) else data
        if rows:
            df = pd.DataFrame(
                {
                    "date": [_parse_date(_pick(r, "EOD_TIMESTAMP", "TIMESTAMP")) for r in rows],
                    "close": [_num(_pick(r, "EOD_CLOSE_INDEX_VAL", "CLOSE")) for r in rows],
                }
            ).dropna()
            return df.drop_duplicates("date").sort_values("date").set_index("date")
        return yfinance_history("^NSEI", start, end)

    def fno_symbols(self) -> set[str]:
        payload = self.get_json("/api/equity-stockIndices", {"index": "SECURITIES IN F&O"}) or {}
        return {
            str(r.get("symbol", "")).upper()
            for r in payload.get("data", [])
            if r.get("symbol") and r.get("priority", 0) == 0
        }

    def lot_size(self, symbol: str) -> int | None:
        payload = self.get_json("/api/quote-derivative", {"symbol": symbol}) or {}
        for s in payload.get("stocks", []):
            lot = s.get("marketDeptOrderBook", {}).get("tradeInfo", {}).get("marketLot") or s.get("metadata", {}).get("marketLot")
            if lot:
                return int(_num(lot))
        return None

    def option_chain(self, symbol: str) -> dict:
        meta = self.get_json("/api/option-chain-contract-info", {"symbol": symbol}) or {}
        expiries = meta.get("expiryDates") or []
        if expiries:
            payload = self.get_json("/api/option-chain-v3", {"type": "Equity", "symbol": symbol, "expiry": expiries[0]})
            if payload and payload.get("records", {}).get("data"):
                payload["records"].setdefault("expiryDates", expiries)
                return payload
        return self.get_json("/api/option-chain-equities", {"symbol": symbol}) or {}

def yfinance_history(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()
    try:
        raw = yf.download(ticker, start=start, end=end + dt.timedelta(days=1), progress=False, auto_adjust=False)
    except Exception as e:
        log.warning("yfinance %s failed: %s", ticker, e)
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = [d.date() for d in df.index]
    df.index.name = "date"
    return df

# ============================================================================
# 2. ANALYSIS MODULE
# ============================================================================
def summarize_deals(deals: pd.DataFrame, min_value_cr: float = 0.0) -> pd.DataFrame:
    if deals.empty:
        return pd.DataFrame()
    d = deals.copy()
    d["qty"] = pd.to_numeric(d["qty"], errors="coerce")
    d["price"] = pd.to_numeric(d["price"], errors="coerce")
    d = d.dropna(subset=["qty", "price"])
    d["value_cr"] = d["qty"] * d["price"] / CRORE
    d["signed_value_cr"] = np.where(d["side"] == "BUY", d["value_cr"], -d["value_cr"])
    d["signed_qty"] = np.where(d["side"] == "BUY", d["qty"], -d["qty"])

    def top_clients(g: pd.DataFrame, side: str) -> str:
        s = g[g["side"] == side].groupby("client")["value_cr"].sum().nlargest(3)
        return "; ".join(f"{c} (₹{v:,.1f} Cr)" for c, v in s.items())

    rows = []
    for sym, g in d.groupby("symbol"):
        buy = g.loc[g["side"] == "BUY", "value_cr"].sum()
        sell = g.loc[g["side"] == "SELL", "value_cr"].sum()
        buy_qty = g.loc[g["side"] == "BUY", "qty"].sum()
        sell_qty = g.loc[g["side"] == "SELL", "qty"].sum()
        rows.append(
            {
                "symbol": sym,
                "name": g["name"].iloc[0],
                "first_deal": g["date"].min(),
                "last_deal": g["date"].max(),
                "deals": len(g),
                "deal_types": ",".join(sorted(g["deal_type"].unique())),
                "buy_value_cr": buy,
                "sell_value_cr": sell,
                "net_value_cr": buy - sell,
                "gross_value_cr": buy + sell,
                "net_qty": g["signed_qty"].sum(),
                "avg_buy_price": (g.loc[g["side"] == "BUY", "qty"] * g.loc[g["side"] == "BUY", "price"]).sum() / buy_qty if buy_qty else np.nan,
                "avg_sell_price": (g.loc[g["side"] == "SELL", "qty"] * g.loc[g["side"] == "SELL", "price"]).sum() / sell_qty if sell_qty else np.nan,
                "top_buyers": top_clients(g, "BUY"),
                "top_sellers": top_clients(g, "SELL"),
            }
        )
    out = pd.DataFrame(rows)
    out["deal_bias"] = np.select(
        [out["net_value_cr"] > 0.1 * out["gross_value_cr"], out["net_value_cr"] < -0.1 * out["gross_value_cr"]],
        ["NET BUY", "NET SELL"],
        "MIXED / CHURN",
    )
    out = out[out["gross_value_cr"] >= min_value_cr]
    return out.sort_values("gross_value_cr", ascending=False).reset_index(drop=True)

def add_indicators(px: pd.DataFrame) -> pd.DataFrame:
    df = px.copy()
    c, h, l = df["close"], df["high"], df["low"]
    df["sma20"] = c.rolling(20).mean()
    df["sma50"] = c.rolling(50).mean()
    df["sma200"] = c.rolling(200).mean()
    df["ema9"] = c.ewm(span=9, adjust=False).mean()
    df["ema21"] = c.ewm(span=21, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = (100 - 100 / (1 + rs)).fillna(100)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["vol_avg20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_avg20"]
    df["swing_low10"] = l.rolling(10).min()
    df["swing_high10"] = h.rolling(10).max()
    df["high52w"] = h.rolling(252, min_periods=20).max()
    df["low52w"] = l.rolling(252, min_periods=20).min()
    return df

def post_deal_behaviour(px: pd.DataFrame, deal_date: dt.date, deal_price: float | None, benchmark: pd.DataFrame | None = None) -> dict:
    if px.empty:
        return {}
    idx = [d for d in px.index if d >= deal_date]
    if not idx:
        return {}
    d0 = idx[0]
    pos = px.index.get_loc(d0)
    base_close = px["close"].iloc[pos]
    prev_close = px["close"].iloc[pos - 1] if pos > 0 else np.nan
    last_close = px["close"].iloc[-1]
    out = {
        "deal_day": d0,
        "deal_day_close": base_close,
        "deal_day_change_pct": (base_close / prev_close - 1) * 100 if prev_close == prev_close else np.nan,
        "deal_day_vol_ratio": px["vol_ratio"].iloc[pos] if "vol_ratio" in px else np.nan,
        "deal_day_delivery_pct": px["delivery_pct"].iloc[pos] if "delivery_pct" in px else np.nan,
        "sessions_since": len(px) - 1 - pos,
        "last_close": last_close,
        "change_since_deal_pct": (last_close / base_close - 1) * 100,
        "max_up_since_pct": (px["high"].iloc[pos:].max() / base_close - 1) * 100,
        "max_down_since_pct": (px["low"].iloc[pos:].min() / base_close - 1) * 100,
    }
    if deal_price and deal_price == deal_price:
        out["vs_deal_price_pct"] = (last_close / deal_price - 1) * 100
    for n in (1, 3, 5, 10):
        if pos + n < len(px):
            out[f"ret_{n}d_pct"] = (px["close"].iloc[pos + n] / base_close - 1) * 100
    if benchmark is not None and not benchmark.empty:
        b = benchmark[benchmark.index >= d0]["close"]
        if len(b) > 1:
            out["benchmark_change_pct"] = (b.iloc[-1] / b.iloc[0] - 1) * 100
            out["relative_strength_pct"] = out["change_since_deal_pct"] - out["benchmark_change_pct"]
    return out

@dataclass
class TrendView:
    score: int
    label: str
    reasons: list[str] = field(default_factory=list)

def trend_view(df: pd.DataFrame, deal_bias: str | None = None, post: dict | None = None) -> TrendView:
    last = df.iloc[-1]
    pts: list[tuple[int, str]] = []

    def add(cond: bool, w: int, yes: str, no: str | None = None) -> None:
        if cond:
            pts.append((w, yes))
        elif no:
            pts.append((-w, no))

    c = last["close"]
    if not np.isnan(last["sma20"]):
        add(c > last["sma20"], 10, "Close above 20-DMA", "Close below 20-DMA")
    if not np.isnan(last["sma50"]):
        add(c > last["sma50"], 12, "Close above 50-DMA", "Close below 50-DMA")
    if not np.isnan(last["sma200"]):
        add(c > last["sma200"], 12, "Close above 200-DMA (long-term uptrend)", "Close below 200-DMA (long-term downtrend)")
    add(last["ema9"] > last["ema21"], 10, "9-EMA above 21-EMA (short-term momentum up)", "9-EMA below 21-EMA (short-term momentum down)")
    add(last["macd"] > last["macd_signal"], 10, "MACD above signal", "MACD below signal")
    rsi = last["rsi14"]
    if rsi >= 75:
        pts.append((-5, f"RSI {rsi:.0f} - overbought, chase risk"))
    elif rsi >= 55:
        pts.append((10, f"RSI {rsi:.0f} - bullish zone"))
    elif rsi <= 25:
        pts.append((5, f"RSI {rsi:.0f} - oversold, bounce possible"))
    elif rsi <= 45:
        pts.append((-10, f"RSI {rsi:.0f} - bearish zone"))
    else:
        pts.append((0, f"RSI {rsi:.0f} - neutral"))
    if deal_bias == "NET BUY":
        pts.append((12, "Large players were net BUYERS in bulk/block deals"))
    elif deal_bias == "NET SELL":
        pts.append((-12, "Large players were net SELLERS in bulk/block deals"))
    if post:
        chg = post.get("change_since_deal_pct")
        if chg is not None and chg == chg:
            add(chg > 0, 12, f"Price up {chg:.1f}% since the deal (market accepting the deal)", f"Price down {abs(chg):.1f}% since the deal (deal absorbed with weakness)")
        rs = post.get("relative_strength_pct")
        if rs is not None and rs == rs:
            add(rs > 0, 7, f"Outperformed NIFTY by {rs:.1f} pts since deal", f"Underperformed NIFTY by {abs(rs):.1f} pts since deal")
        vr = post.get("deal_day_vol_ratio")
        if vr is not None and vr == vr and vr > 2:
            pts.append((0, f"Deal-day volume {vr:.1f}x the 20-day average"))
    
    score = int(max(-100, min(100, sum(w for w, _ in pts))))
    if score >= 40:
        label = "BULLISH"
    elif score >= 15:
        label = "MILDLY BULLISH"
    elif score > -15:
        label = "NEUTRAL / SIDEWAYS"
    elif score > -40:
        label = "MILDLY BEARISH"
    else:
        label = "BEARISH"
    return TrendView(score, label, [r for _, r in pts])

# ============================================================================
# 3. DERIVATIVES MODULE
# ============================================================================
def _ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_price(spot: float, strike: float, years: float, iv: float, kind: str, r: float = RISK_FREE) -> float:
    if years <= 0 or iv <= 0:
        return max(0.0, spot - strike) if kind == "CE" else max(0.0, strike - spot)
    d1 = (math.log(spot / strike) + (r + iv * iv / 2) * years) / (iv * math.sqrt(years))
    d2 = d1 - iv * math.sqrt(years)
    if kind == "CE":
        return spot * _ncdf(d1) - strike * math.exp(-r * years) * _ncdf(d2)
    return strike * math.exp(-r * years) * _ncdf(-d2) - spot * _ncdf(-d1)

def bs_delta(spot: float, strike: float, years: float, iv: float, kind: str, r: float = RISK_FREE) -> float:
    if years <= 0 or iv <= 0:
        return (1.0 if spot > strike else 0.0) if kind == "CE" else (-1.0 if spot < strike else 0.0)
    d1 = (math.log(spot / strike) + (r + iv * iv / 2) * years) / (iv * math.sqrt(years))
    return _ncdf(d1) if kind == "CE" else _ncdf(d1) - 1

@dataclass
class ChainStats:
    expiry: str
    spot: float
    pcr_oi: float
    max_pain: float
    call_wall: float
    put_wall: float
    call_oi_build: float
    put_oi_build: float
    atm_strike: float
    atm_iv: float
    strike_step: float
    chain: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    @property
    def bias(self) -> str:
        if self.pcr_oi >= 1.2:
            return "BULLISH (heavy put writing - writers defending downside)"
        if self.pcr_oi >= 0.9:
            return "MILDLY BULLISH / NEUTRAL"
        if self.pcr_oi >= 0.6:
            return "MILDLY BEARISH / NEUTRAL"
        return "BEARISH (heavy call writing - writers capping upside)"

def parse_option_chain(payload: dict, expiry: str | None = None) -> ChainStats | None:
    rec = (payload or {}).get("records") or {}
    data = rec.get("data") or []
    if not data:
        return None
    expiries = rec.get("expiryDates") or sorted({r.get("expiryDate") or r.get("expiryDates") for r in data})
    expiry = expiry or expiries[0]
    rows = []
    for r in data:
        e = r.get("expiryDate") or r.get("expiryDates")
        if e != expiry:
            continue
        ce, pe = r.get("CE") or {}, r.get("PE") or {}
        rows.append(
            {
                "strike": float(r["strikePrice"]),
                "ce_oi": float(ce.get("openInterest") or 0),
                "ce_chg_oi": float(ce.get("changeinOpenInterest") or 0),
                "ce_iv": float(ce.get("impliedVolatility") or 0),
                "ce_ltp": float(ce.get("lastPrice") or 0),
                "pe_oi": float(pe.get("openInterest") or 0),
                "pe_chg_oi": float(pe.get("changeinOpenInterest") or 0),
                "pe_iv": float(pe.get("impliedVolatility") or 0),
                "pe_ltp": float(pe.get("lastPrice") or 0),
            }
        )
    if not rows:
        return None
    ch = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
    spot = float(rec.get("underlyingValue") or ch.loc[(ch["ce_oi"] + ch["pe_oi"]).idxmax(), "strike"])
    strikes = ch["strike"].values
    pain = [
        ((s - ch["strike"]).clip(lower=0) * ch["ce_oi"]).sum() + ((ch["strike"] - s).clip(lower=0) * ch["pe_oi"]).sum()
        for s in strikes
    ]
    max_pain = float(strikes[int(pd.Series(pain).idxmin())])
    atm_i = int((ch["strike"] - spot).abs().idxmin())
    atm = ch.loc[atm_i]
    ivs = [v for v in (atm["ce_iv"], atm["pe_iv"]) if v > 0]
    step = float(pd.Series(strikes).diff().median()) if len(strikes) > 1 else spot * 0.01
    return ChainStats(
        expiry=expiry,
        spot=spot,
        pcr_oi=ch["pe_oi"].sum() / ch["ce_oi"].sum() if ch["ce_oi"].sum() else float("nan"),
        max_pain=max_pain,
        call_wall=float(ch.loc[ch["ce_oi"].idxmax(), "strike"]),
        put_wall=float(ch.loc[ch["pe_oi"].idxmax(), "strike"]),
        call_oi_build=float(ch.loc[ch["ce_chg_oi"].idxmax(), "strike"]),
        put_oi_build=float(ch.loc[ch["pe_chg_oi"].idxmax(), "strike"]),
        atm_strike=float(atm["strike"]),
        atm_iv=(sum(ivs) / len(ivs) / 100) if ivs else 0.0,
        strike_step=step,
        chain=ch,
    )

def years_to_expiry(expiry: str, today: dt.date | None = None) -> float:
    today = today or dt.date.today()
    try:
        e = dt.datetime.strptime(expiry, "%d-%b-%Y").date()
    except ValueError:
        return 7 / 365
    return max((e - today).days, 0.5) / 365

@dataclass
class TradePlan:
    direction: str
    rationale: str
    entry: float | None = None
    entry_zone: tuple[float, float] | None = None
    stop_loss: float | None = None
    target1: float | None = None
    target2: float | None = None
    trailing_rule: str = ""
    risk_per_share: float | None = None
    reward_risk_t1: float | None = None
    futures: dict = field(default_factory=dict)
    options: list[dict] = field(default_factory=list)
    invalidation: str = ""

def build_trade_plan(df: pd.DataFrame, trend_score: int, chain: ChainStats | None, lot_size: int | None,
                     fno_eligible: bool, capital: float = 500000.0, risk_pct: float = 1.0, today: dt.date | None = None) -> TradePlan:
    last = df.iloc[-1]
    close, atr = float(last["close"]), float(last["atr14"])
    if trend_score >= 15:
        direction = "LONG"
    elif trend_score <= -15:
        direction = "SHORT"
        if not fno_eligible:
            return TradePlan(
                "AVOID / EXIT LONGS",
                f"Trend score {trend_score} is bearish, but the stock is not in F&O so it cannot be shorted "
                "overnight (cash-market shorts must be squared off intraday). Avoid fresh longs; holders can "
                f"exit or trail a stop above {last['swing_high10']:.2f}.",
                invalidation=f"Bearish view fails on a daily close above {last['swing_high10']:.2f}",
            )
    else:
        return TradePlan(
            "NO TRADE",
            f"Trend score {trend_score} is inside the -15..+15 neutral band - no directional edge. "
            "Wait for a breakout above the 10-day high or breakdown below the 10-day low.",
            invalidation=f"Watch {last['swing_high10']:.2f} (breakout) / {last['swing_low10']:.2f} (breakdown)",
        )

    if direction == "LONG":
        entry_lo, entry_hi = close - 0.3 * atr, close + 0.2 * atr
        entry = close
        sl = min(float(last["swing_low10"]), close - 1.5 * atr)
        if chain and sl - 0.75 * atr < chain.put_wall < close:
            sl = min(sl, chain.put_wall - 0.25 * atr)
        sl = max(sl, close - 3 * atr)
        risk = entry - sl
        t1, t2 = entry + 1.5 * risk, entry + 3 * risk
        if chain and entry + risk <= chain.call_wall < t1:
            t1 = chain.call_wall
    else:
        entry_lo, entry_hi = close - 0.2 * atr, close + 0.3 * atr
        entry = close
        sl = max(float(last["swing_high10"]), close + 1.5 * atr)
        if chain and close < chain.call_wall < sl + 0.75 * atr:
            sl = max(sl, chain.call_wall + 0.25 * atr)
        sl = min(sl, close + 3 * atr)
        risk = sl - entry
        t1, t2 = entry - 1.5 * risk, entry - 3 * risk
        if chain and t1 < chain.put_wall <= entry - risk:
            t1 = chain.put_wall

    plan = TradePlan(
        direction=direction,
        rationale=f"Trend score {trend_score:+d}. ATR(14) = {atr:.2f}.",
        entry=round(entry, 2),
        entry_zone=(round(entry_lo, 2), round(entry_hi, 2)),
        stop_loss=round(sl, 2),
        target1=round(t1, 2),
        target2=round(t2, 2),
        trailing_rule="After T1, book 50% and move the stop to entry (cost). Trail the rest with the 21-EMA on a closing basis.",
        risk_per_share=round(risk, 2),
        reward_risk_t1=round(abs(t1 - entry) / risk, 2) if risk else None,
        invalidation=f"Daily close {'below' if direction == 'LONG' else 'above'} {sl:.2f} invalidates the view.",
    )
    risk_budget = capital * risk_pct / 100
    if fno_eligible and lot_size:
        lots_by_risk = int(risk_budget // (risk * lot_size)) if risk else 0
        plan.futures = {
            "instrument": "Near-month stock future",
            "side": "BUY" if direction == "LONG" else "SELL",
            "lot_size": lot_size,
            "risk_per_lot": round(risk * lot_size, 0),
            "reward_t1_per_lot": round(abs(t1 - entry) * lot_size, 0),
            "approx_contract_value": round(entry * lot_size, 0),
            "approx_margin_(~20%)": round(entry * lot_size * 0.20, 0),
            "lots_for_risk_budget": lots_by_risk,
            "risk_budget": round(risk_budget, 0),
            "note": "Futures trade at a premium/discount to spot; apply the same point distances to the futures price at entry. Margin is an estimate - check your broker's SPAN+exposure."
                    + (" One lot risks more than your budget - use options or skip." if lots_by_risk == 0 else ""),
        }
    if fno_eligible and chain:
        plan.options = option_ideas(direction, entry, sl, t1, chain, lot_size, risk_budget, today)
    return plan

def option_ideas(direction: str, entry: float, sl: float, t1: float, chain: ChainStats,
                 lot_size: int | None, risk_budget: float, today: dt.date | None) -> list[dict]:
    T = years_to_expiry(chain.expiry, today)
    iv = chain.atm_iv or 0.30
    kind = "CE" if direction == "LONG" else "PE"
    step = chain.strike_step
    atm = chain.atm_strike
    otm = atm + step if kind == "CE" else atm - step
    ideas = []
    T_exit = max(T - 3 / 365, 0.5 / 365)
    for label, strike in (("ATM", atm), ("1 strike OTM", otm)):
        row = chain.chain[chain.chain["strike"] == strike]
        ltp = float(row[f"{kind.lower()}_ltp"].iloc[0]) if not row.empty else 0.0
        model_now = bs_price(entry, strike, T, iv, kind)
        prem = ltp if ltp > 0 else model_now
        prem_sl = bs_price(sl, strike, T_exit, iv, kind)
        prem_t1 = bs_price(t1, strike, T_exit, iv, kind)
        prem_sl = max(prem_sl, prem * 0.6)
        risk_lot = (prem - prem_sl) * (lot_size or 0)
        ideas.append(
            {
                "strategy": f"Buy {label} {kind}",
                "strike": strike,
                "expiry": chain.expiry,
                "premium_entry": round(prem, 2),
                "premium_source": "NSE LTP" if ltp > 0 else "Black-Scholes estimate",
                "premium_stop_loss": round(prem_sl, 2),
                "premium_target": round(prem_t1, 2),
                "delta": round(bs_delta(entry, strike, T, iv, kind), 2),
                "risk_per_lot": round(risk_lot, 0) if lot_size else None,
                "lots_for_risk_budget": int(risk_budget // risk_lot) if lot_size and risk_lot > 0 else None,
                "exit_rule": "Exit the option if the UNDERLYING closes beyond the stop-loss, even if the premium stop has not hit.",
            }
        )
    far = otm + 2 * step if kind == "CE" else otm - 2 * step
    buy_p = bs_price(entry, atm, T, iv, kind)
    sell_p = bs_price(entry, far, T, iv, kind)
    width = abs(far - atm)
    ideas.append(
        {
            "strategy": f"Bull call spread (buy {atm:g} CE / sell {far:g} CE)" if kind == "CE" else f"Bear put spread (buy {atm:g} PE / sell {far:g} PE)",
            "expiry": chain.expiry,
            "net_debit_est": round(buy_p - sell_p, 2),
            "max_profit_est": round(width - (buy_p - sell_p), 2),
            "max_loss_est": round(buy_p - sell_p, 2),
            "when": f"Prefer this over naked buying when ATM IV is high (current ATM IV ≈ {iv * 100:.0f}%) or expiry is under 10 days away.",
        }
    )
    sell_kind = "PE" if kind == "CE" else "CE"
    wall = chain.put_wall if sell_kind == "PE" else chain.call_wall
    ideas.append(
        {
            "strategy": f"Sell {wall:g} {sell_kind} (writer view, needs full margin)",
            "expiry": chain.expiry,
            "why": f"Highest {sell_kind} open interest sits at {wall:g} - writers are defending it.",
            "stop": f"Exit if the underlying closes {'below' if sell_kind == 'PE' else 'above'} {wall:g}.",
            "warning": "Unlimited-risk position. Only with a hedge or strict stop.",
        }
    )
    return ideas

# ============================================================================
# 4. SENTIMENT MODULE (GEMINI)
# ============================================================================
def research_sentiment(symbol: str, company: str, deal_context: str, technical_context: str,
                       client=None, max_searches: int = 6) -> dict:
    """Returns the parsed JSON dict, or {"error": ...} if Gemini is unavailable."""
    try:
        import google.generativeai as genai
    except ImportError:
        return {"error": "google-generativeai package not installed - run `pip install google-generativeai`"}
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY environment variable not set. Please set it to use AI sentiment analysis."}
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            MODEL,
            system_instruction=SYSTEM
        )
        
        user_prompt = (
            f"Stock: {company} (NSE: {symbol})\n\n"
            f"Bulk/block deal activity:\n{deal_context}\n\n"
            f"Price/technical context computed from exchange data:\n{technical_context}\n\n"
            "Analyze the news flow and sentiment for this stock now, explain the deal, and return the JSON."
        )
        
        response = model.generate_content(
            user_prompt,
            generation_config={
                "temperature": 0.2,
                "response_mime_type": "application/json"
            }
        )
        
        parsed = json.loads(response.text)
        return parsed
        
    except json.JSONDecodeError:
        return {"error": "Could not parse sentiment JSON", "raw": response.text[-2000:]}
    except Exception as e:
        return {"error": f"Gemini API error: {str(e)}"}

# ============================================================================
# 5. DEMO MODULE
# ============================================================================
def _bdays(end: dt.date, n: int) -> list[dt.date]:
    return [d.date() for d in pd.bdate_range(end=end, periods=n)]

def _walk(seed: int, n: int, start: float, drift: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.016, n)
    close = start * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0.002, 0.018, n))
    low = close * (1 - rng.uniform(0.002, 0.018, n))
    open_ = np.clip(close * (1 + rng.normal(0, 0.006, n)), low, high)
    vol = rng.integers(400_000, 1_600_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol, "delivery_pct": rng.uniform(30, 65, n)})

class DemoNSE:
    STOCKS = {"ALPHAINFRA": (11, 820.0, 0.0018), "BETAPHARMA": (23, 1460.0, -0.0016), "GAMMASMALL": (37, 96.0, 0.0002)}

    def __init__(self, today: dt.date | None = None):
        self.today = today or dt.date.today()
        self.dates = _bdays(self.today, 320)

    def deals(self, start, end, kind="bulk"):
        if kind != "bulk":
            return normalize_deal_rows([], "BLOCK")
        d = self.dates[-8].strftime("%d-%b-%Y")
        rows = [
            ("ALPHAINFRA", "Alpha Infra Ltd", "GOLDMAN SACHS (SINGAPORE) PTE", "BUY", 1_800_000, self._px("ALPHAINFRA", -8)),
            ("ALPHAINFRA", "Alpha Infra Ltd", "PROMOTER HOLDCO LLP", "SELL", 400_000, self._px("ALPHAINFRA", -8)),
            ("BETAPHARMA", "Beta Pharma Ltd", "XYZ PRIVATE EQUITY FUND IV", "SELL", 2_100_000, self._px("BETAPHARMA", -8)),
            ("BETAPHARMA", "Beta Pharma Ltd", "SBI MUTUAL FUND", "BUY", 600_000, self._px("BETAPHARMA", -8)),
            ("GAMMASMALL", "Gamma Small Caps Ltd", "RANDOM TRADER", "BUY", 3_000_000, self._px("GAMMASMALL", -8)),
            ("GAMMASMALL", "Gamma Small Caps Ltd", "RANDOM TRADER", "SELL", 2_950_000, self._px("GAMMASMALL", -8)),
        ]
        return normalize_deal_rows(
            [{"BD_DT_DATE": d, "BD_SYMBOL": s, "BD_SCRIP_NAME": n, "BD_CLIENT_NAME": c, "BD_BUY_SELL": side,
              "BD_QTY_TRD": q, "BD_TP_WATP": round(p, 2)} for s, n, c, side, q, p in rows], "BULK")

    def latest_large_deals(self):
        return self.deals(None, None)

    def _frame(self, sym):
        seed, start, drift = self.STOCKS[sym]
        df = _walk(seed, len(self.dates), start, drift)
        df.index = self.dates
        df.index.name = "date"
        df.loc[df.index[-8], "volume"] *= 4
        return df

    def _px(self, sym, i):
        return float(self._frame(sym)["close"].iloc[i])

    def price_history(self, symbol, start, end):
        return self._frame(symbol) if symbol in self.STOCKS else pd.DataFrame()

    def index_history(self, start, end):
        df = _walk(5, len(self.dates), 25000.0, 0.0004)[["close"]]
        df.index = self.dates
        return df[df.index >= start]

    def fno_symbols(self):
        return {"ALPHAINFRA", "BETAPHARMA"}

    def lot_size(self, symbol):
        return {"ALPHAINFRA": 700, "BETAPHARMA": 350}.get(symbol)

    def option_chain(self, symbol):
        spot = float(self._frame(symbol)["close"].iloc[-1])
        step = 20.0 if spot < 1000 else 50.0
        atm = round(spot / step) * step
        expiry = (self.today + dt.timedelta(days=(3 - self.today.weekday()) % 7 + 14)).strftime("%d-%b-%Y")
        bullish = symbol == "ALPHAINFRA"
        data = []
        for k in range(-8, 9):
            strike = atm + k * step
            ce_oi = 2000 + (6000 if k == (3 if bullish else 1) else 0) + max(0, k) * 300
            pe_oi = 2000 + (7000 if k == (-1 if bullish else -3) else 0) + max(0, -k) * 300
            data.append({"strikePrice": strike, "expiryDate": expiry,
                         "CE": {"openInterest": ce_oi, "changeinOpenInterest": ce_oi * 0.1,
                                "impliedVolatility": 28 + abs(k), "lastPrice": 0},
                         "PE": {"openInterest": pe_oi, "changeinOpenInterest": pe_oi * 0.1,
                                "impliedVolatility": 30 + abs(k), "lastPrice": 0}})
        return {"records": {"data": data, "expiryDates": [expiry], "underlyingValue": spot}}

# ============================================================================
# 6. AGENT MODULE
# ============================================================================
@dataclass
class StockReport:
    symbol: str
    deal_summary: dict
    post_deal: dict
    trend: TrendView
    latest: dict
    fno_eligible: bool
    lot_size: int | None
    chain: ChainStats | None
    sentiment: dict
    final_score: int
    final_label: str
    plan: TradePlan
    notes: list[str] = field(default_factory=list)

@dataclass
class Config:
    start: dt.date
    end: dt.date
    min_value_cr: float = 10.0
    top_n: int = 10
    symbols: list[str] | None = None
    capital: float = 500000.0
    risk_pct: float = 1.0
    use_claude: bool = True  # Kept for backward compatibility, now drives Gemini
    include_block: bool = True

def label_for(score: int) -> str:
    if score >= 40:
        return "BULLISH"
    if score >= 15:
        return "MILDLY BULLISH"
    if score > -15:
        return "NEUTRAL / SIDEWAYS"
    if score > -40:
        return "MILDLY BEARISH"
    return "BEARISH"

class DealResearchAgent:
    def __init__(self, cfg: Config, nse: NSEClient | None = None, claude_client=None):
        self.cfg = cfg
        self.nse = nse or NSEClient()
        self.claude_client = claude_client

    def collect_deals(self, deals: pd.DataFrame | None = None) -> pd.DataFrame:
        if deals is None:
            frames = [self.nse.deals(self.cfg.start, self.cfg.end, "bulk")]
            if self.cfg.include_block:
                frames.append(self.nse.deals(self.cfg.start, self.cfg.end, "block"))
            deals = pd.concat(frames, ignore_index=True)
            if deals.empty and self.cfg.end >= dt.date.today() - dt.timedelta(days=1):
                deals = self.nse.latest_large_deals()
        if self.cfg.symbols:
            deals = deals[deals["symbol"].isin([s.upper() for s in self.cfg.symbols])]
        return deals

    def analyse_symbol(self, row: dict, fno: set[str], benchmark: pd.DataFrame | None) -> StockReport | None:
        sym = row["symbol"]
        notes: list[str] = []
        hist_start = row["first_deal"] - dt.timedelta(days=400)
        px = self.nse.price_history(sym, hist_start, dt.date.today())
        if px.empty or len(px) < 30:
            log.warning("%s: not enough price history, skipping", sym)
            return None
        
        px = add_indicators(px)
        deal_px = row["avg_buy_price"] if row["deal_bias"] == "NET BUY" else row["avg_sell_price"]
        post = post_deal_behaviour(px, row["first_deal"], deal_px, benchmark)
        trend = trend_view(px, row["deal_bias"], post)
        last = px.iloc[-1]
        latest = {k: float(last[k]) for k in ("close", "sma20", "sma50", "sma200", "rsi14", "atr14",
                                              "macd", "macd_signal", "swing_low10", "swing_high10",
                                              "high52w", "low52w")}
        latest["date"] = px.index[-1]
        eligible = sym in fno
        lot = chain = None
        if eligible:
            lot = self.nse.lot_size(sym)
            chain = parse_option_chain(self.nse.option_chain(sym))
            if lot is None:
                notes.append("Lot size unavailable from NSE - check the F&O contract specs.")
            if chain is None:
                notes.append("Option chain unavailable - options section skipped.")
        else:
            notes.append("Not in NSE's F&O list - only cash-market (delivery/intraday) trades are possible.")
        
        senti: dict = {}
        if self.cfg.use_claude:
            senti = research_sentiment(
                sym, row.get("name") or sym,
                deal_context=_deal_context(row, post),
                technical_context=_tech_context(latest, trend, chain),
                client=self.claude_client,
            )
            if "error" in senti:
                notes.append(f"Sentiment research skipped: {senti['error']}")
        
        s_score = senti.get("sentiment_score") if isinstance(senti.get("sentiment_score"), (int, float)) else None
        final = trend.score if s_score is None else round(0.6 * trend.score + 0.4 * s_score)
        plan = build_trade_plan(px, final, chain, lot, eligible, self.cfg.capital, self.cfg.risk_pct)
        return StockReport(sym, row, post, trend, latest, eligible, lot, chain, senti, final,
                           label_for(final), plan, notes)

    def run(self, deals: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[StockReport]]:
        deals = self.collect_deals(deals)
        summary = summarize_deals(deals, self.cfg.min_value_cr)
        if summary.empty:
            return summary, []
        fno = self.nse.fno_symbols()
        benchmark = self.nse.index_history(self.cfg.start - dt.timedelta(days=10), dt.date.today())
        reports = []
        
        for row in summary.head(self.cfg.top_n).to_dict("records"):
            log.info("Analysing %s", row["symbol"])
            r = self.analyse_symbol(row, fno, benchmark)
            if r:
                reports.append(r)
            time.sleep(2) # QUOTA SAFEGUARD: Prevent hitting the 15 RPM limit on Gemini Free Tier
            
        return summary, reports

def _deal_context(row: dict, post: dict) -> str:
    lines = [
        f"Window: {row['first_deal']} to {row['last_deal']}; {row['deals']} deal(s) ({row['deal_types']}).",
        f"Bought ₹{row['buy_value_cr']:,.1f} Cr, sold ₹{row['sell_value_cr']:,.1f} Cr -> {row['deal_bias']}.",
    ]
    if row.get("top_buyers"):
        lines.append(f"Top buyers: {row['top_buyers']}")
    if row.get("top_sellers"):
        lines.append(f"Top sellers: {row['top_sellers']}")
    if post:
        lines.append(
            f"Since the deal: {post.get('change_since_deal_pct', float('nan')):+.1f}% over "
            f"{post.get('sessions_since')} sessions; deal-day move {post.get('deal_day_change_pct', float('nan')):+.1f}%."
        )
    return "\n".join(lines)

def _tech_context(latest: dict, trend: TrendView, chain) -> str:
    s = (f"Close {latest['close']:.2f}, RSI {latest['rsi14']:.0f}, 20/50/200-DMA "
         f"{latest['sma20']:.2f}/{latest['sma50']:.2f}/{latest['sma200']:.2f}. "
         f"Rule-based trend: {trend.label} ({trend.score:+d}).")
    if chain:
        s += (f" Options: PCR {chain.pcr_oi:.2f}, max pain {chain.max_pain:g}, call wall {chain.call_wall:g}, "
              f"put wall {chain.put_wall:g}, ATM IV {chain.atm_iv * 100:.0f}%.")
    return s

# ============================================================================
# 7. REPORT MODULE
# ============================================================================
def _f(v, fmt="{:,.2f}", na="—"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return na
    return fmt.format(v)

def _v(v):
    if isinstance(v, float):
        return f"{v:,.2f}" if abs(v) < 1000 else f"{v:,.0f}"
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}"
    return v

def render(summary: pd.DataFrame, reports: list[StockReport], start: dt.date, end: dt.date) -> str:
    out = [f"# NSE Bulk & Block Deal Research — {start:%d %b %Y} to {end:%d %b %Y}", "",
           f"_Generated {dt.datetime.now():%d %b %Y %H:%M}_", "", DISCLAIMER, ""]
    if summary.empty:
        out.append("No bulk/block deals matched the filters.")
        return "\n".join(out)
    
    out += ["## Deal scan", "",
            "| Symbol | Deals | Buy ₹Cr | Sell ₹Cr | Net ₹Cr | Bias |",
            "|---|---:|---:|---:|---:|---|"]
    for r in summary.itertuples():
        out.append(f"| {r.symbol} | {r.deals} | {r.buy_value_cr:,.1f} | {r.sell_value_cr:,.1f} | "
                   f"{r.net_value_cr:+,.1f} | {r.deal_bias} |")
    out.append("")
    
    if reports:
        out += ["## Verdict board", "",
                "| Symbol | Since deal | Technical | News | Final view | F&O | Trade |",
                "|---|---:|---:|---|---|---|---|"]
        for r in reports:
            news = r.sentiment.get("sentiment", "—") if r.sentiment and "error" not in r.sentiment else "—"
            trade = r.plan.direction
            if r.plan.entry:
                trade += f" @ {r.plan.entry:,.2f}, SL {r.plan.stop_loss:,.2f}, T1 {r.plan.target1:,.2f}"
            out.append(f"| {r.symbol} | {_f(r.post_deal.get('change_since_deal_pct'), '{:+.1f}%')} | "
                       f"{r.trend.score:+d} | {news} | **{r.final_label}** ({r.final_score:+d}) | "
                       f"{'Yes' if r.fno_eligible else 'No'} | {trade} |")
        out.append("")
    
    for r in reports:
        out += _stock_section(r)
    return "\n".join(out)

def _stock_section(r: StockReport) -> list[str]:
    d, p, L = r.deal_summary, r.post_deal, r.latest
    o = [f"---", f"## {r.symbol} — {d.get('name') or ''}", "",
         f"**Final view: {r.final_label} ({r.final_score:+d}/100)**", "", "### 1. The deal", "",
         f"- {d['deals']} {d['deal_types'].lower()} deal(s) between {d['first_deal']} and {d['last_deal']}",
         f"- Bought ₹{d['buy_value_cr']:,.1f} Cr (avg {_f(d['avg_buy_price'])}), sold ₹{d['sell_value_cr']:,.1f} Cr "
         f"(avg {_f(d['avg_sell_price'])}) → **{d['deal_bias']}**"]
    if d.get("top_buyers"):
        o.append(f"- Buyers: {d['top_buyers']}")
    if d.get("top_sellers"):
        o.append(f"- Sellers: {d['top_sellers']}")
    
    o += ["", "### 2. How the stock behaved after the deal", ""]
    if p:
        o += [
            f"- Deal day ({p['deal_day']}): close {_f(p['deal_day_close'])}, move {_f(p.get('deal_day_change_pct'), '{:+.2f}%')}, "
            f"volume {_f(p.get('deal_day_vol_ratio'), '{:.1f}x')} of 20-day avg, delivery {_f(p.get('deal_day_delivery_pct'), '{:.0f}%')}",
            f"- Returns after deal: 1d {_f(p.get('ret_1d_pct'), '{:+.1f}%')}, 3d {_f(p.get('ret_3d_pct'), '{:+.1f}%')}, "
            f"5d {_f(p.get('ret_5d_pct'), '{:+.1f}%')}, 10d {_f(p.get('ret_10d_pct'), '{:+.1f}%')}",
            f"- Now {_f(p['last_close'])} → **{_f(p['change_since_deal_pct'], '{:+.1f}%')}** over {p['sessions_since']} sessions "
            f"(best {_f(p['max_up_since_pct'], '{:+.1f}%')}, worst {_f(p['max_down_since_pct'], '{:+.1f}%')})",
        ]
        if "vs_deal_price_pct" in p:
            o.append(f"- Vs the deal price: {_f(p['vs_deal_price_pct'], '{:+.1f}%')} "
                     f"({'big players in profit' if p['vs_deal_price_pct'] > 0 else 'big players under water'})")
        if "relative_strength_pct" in p:
            o.append(f"- NIFTY moved {_f(p['benchmark_change_pct'], '{:+.1f}%')} in the same period → relative strength "
                     f"{_f(p['relative_strength_pct'], '{:+.1f} pts')}")
    else:
        o.append("- No sessions since the deal yet.")
    
    o += ["", f"### 3. Technical trend: {r.trend.label} ({r.trend.score:+d})", "",
          f"Close {_f(L['close'])} | 20-DMA {_f(L['sma20'])} | 50-DMA {_f(L['sma50'])} | 200-DMA {_f(L['sma200'])} | "
          f"RSI {_f(L['rsi14'], '{:.0f}')} | ATR {_f(L['atr14'])} | 52w {_f(L['low52w'])}–{_f(L['high52w'])}", ""]
    o += [f"- {x}" for x in r.trend.reasons]
    o += ["", "### 4. News & market sentiment", ""]
    
    s = r.sentiment
    if s and "error" not in s:
        o.append(f"**{s.get('sentiment', '?')} ({s.get('sentiment_score', '?')})** — {s.get('summary', '')}")
        if s.get("deal_interpretation"):
            o += ["", f"_Deal interpretation:_ {s['deal_interpretation']}"]
        for title, key in (("Drivers", "key_drivers"), ("Upcoming events", "upcoming_events"), ("Risks", "risks")):
            if s.get(key):
                o += ["", f"**{title}:**"] + [f"- {x}" for x in s[key]]
        if s.get("headlines"):
            o += ["", "**Headlines:**"]
            for h in s["headlines"][:8]:
                o.append(f"- {h.get('date', '')} — [{h.get('title', '')}]({h.get('url', '')}) ({h.get('source', '')})")
    else:
        o.append("_Not run._" if not s else f"_Unavailable: {s.get('error')}_")
    
    o += ["", "### 5. F&O analysis", ""]
    if not r.fno_eligible:
        o.append("Not in the F&O segment — futures/options trading is not possible on this stock.")
    else:
        o.append(f"F&O eligible. Lot size: **{r.lot_size or 'n/a'}**.")
        c = r.chain
        if c:
            o += ["", f"- Expiry {c.expiry}; spot {_f(c.spot)}; ATM {c.atm_strike:g}; ATM IV {c.atm_iv * 100:.0f}%",
                  f"- PCR (OI) {c.pcr_oi:.2f} → {c.bias}",
                  f"- Max pain {c.max_pain:g}",
                  f"- Resistance (highest call OI) {c.call_wall:g}; fresh call writing at {c.call_oi_build:g}",
                  f"- Support (highest put OI) {c.put_wall:g}; fresh put writing at {c.put_oi_build:g}"]
    
    pl = r.plan
    o += ["", f"### 6. Trade plan: {pl.direction}", "", pl.rationale]
    if pl.entry:
        o += ["", "| | Level |", "|---|---:|",
              f"| Entry | {pl.entry:,.2f} (zone {pl.entry_zone[0]:,.2f}–{pl.entry_zone[1]:,.2f}) |",
              f"| Stop-loss | {pl.stop_loss:,.2f} (risk {pl.risk_per_share:,.2f}/share) |",
              f"| Target 1 | {pl.target1:,.2f} (R:R {pl.reward_risk_t1}) |",
              f"| Target 2 | {pl.target2:,.2f} |", "",
              f"- {pl.trailing_rule}", f"- {pl.invalidation}"]
        if pl.futures:
            o += ["", "**Futures:**"] + [f"- {k.replace('_', ' ')}: {_v(v)}" for k, v in pl.futures.items()]
        if pl.options:
            o += ["", "**Options:**"]
            for idea in pl.options:
                o.append(f"- **{idea['strategy']}** — " + "; ".join(
                    f"{k.replace('_', ' ')}: {_v(v)}" for k, v in idea.items() if k != "strategy"))
    else:
        o += ["", f"- {pl.invalidation}"]
    
    if r.notes:
        o += ["", "_Notes:_"] + [f"- {n}" for n in r.notes]
    o.append("")
    return o

# ============================================================================
# 8. MAIN CLI MODULE (LIVE DATA ENABLED)
# ============================================================================
def _date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="nse_deal_agent",
        description="Research NSE bulk/block deals, post-deal price action, news sentiment and F&O trade setups.",
    )
    p.add_argument("--from", dest="start", type=_date, help="start date YYYY-MM-DD (default: 7 days ago)")
    p.add_argument("--to", dest="end", type=_date, help="end date YYYY-MM-DD (default: today)")
    p.add_argument("--symbols", nargs="*", help="only analyse these NSE symbols")
    p.add_argument("--min-value", type=float, default=10.0, help="minimum gross deal value in ₹ crore (default 10)")
    p.add_argument("--top", type=int, default=3, help="analyse the N largest deals (default 3 to protect free AI quota)")
    p.add_argument("--capital", type=float, default=500000, help="trading capital in ₹ for position sizing")
    p.add_argument("--risk-pct", type=float, default=1.0, help="%% of capital to risk per trade (default 1)")
    p.add_argument("--deals-csv", nargs="*", help="NSE bulk/block deal CSV(s) downloaded manually")
    p.add_argument("--bulk-only", action="store_true", help="skip block deals")
    p.add_argument("--no-ai", action="store_true", help="skip AI news/sentiment research")
    p.add_argument("--demo", action="store_true", help="run on synthetic data (no network)")
    p.add_argument("--out", default="nse_deal_report.md", help="markdown report path")
    p.add_argument("-v", "--verbose", action="store_true")
    
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    
    end = a.end or dt.date.today()
    start = a.start or end - dt.timedelta(days=7)
    use_ai = not a.no_ai and not a.demo
    
    if use_ai and not os.environ.get("GEMINI_API_KEY"):
        print("note: no GEMINI_API_KEY set - use --no-ai to skip news research", file=sys.stderr)
    
    cfg = Config(start=start, end=end, min_value_cr=a.min_value, top_n=a.top, symbols=a.symbols,
                 capital=a.capital, risk_pct=a.risk_pct, use_claude=use_ai, include_block=not a.bulk_only)
    
    if a.demo:
        log.info("🎭 Running in DEMO mode with synthetic data.")
        nse = DemoNSE()
    else:
        log.info("🌐 Running in LIVE mode. Initializing secure NSE session...")
        nse = NSEClient()
    
    deals = None
    if a.deals_csv:
        try:
            deals = pd.concat([load_deals_csv(f, "BLOCK" if "block" in f.lower() else "BULK") for f in a.deals_csv],
                              ignore_index=True)
            deals = deals[(deals["date"] >= start) & (deals["date"] <= end)]
            log.info(f"✅ Loaded {len(deals)} deals from provided CSV file(s).")
        except Exception as e:
            log.error(f"❌ Failed to load CSV files: {e}")
            deals = pd.DataFrame()
    else:
        try:
            log.info(f"Fetching live deals (Min Value: ₹{a.min_value}Cr) from {start} to {end}...")
            bulk_df = nse.deals(start, end, kind="bulk")
            if not a.bulk_only:
                block_df = nse.deals(start, end, kind="block")
                deals = pd.concat([bulk_df, block_df], ignore_index=True)
            else:
                deals = bulk_df
                
            if deals.empty:
                log.warning("⚠️ NSE APIs returned no data. This may indicate temporary API downtime, weekend markets, or strict WAF blocking.")
            else:
                log.info(f"✅ Successfully fetched {len(deals)} live deals from NSE.")
        except Exception as e:
            log.error(f"❌ Critical error fetching live NSE data: {e}")
            deals = pd.DataFrame()

    if deals is None or deals.empty:
        log.warning("⚠️ No deal data available to analyze. Generating empty report.")
        summary = pd.DataFrame()
        reports = []
    else:
        summary, reports = DealResearchAgent(cfg, nse).run(deals)
        
    md = render(summary, reports, start, end)
    
    try:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(md)
        log.info(f"💾 Report successfully written to: {a.out}")
    except IOError as e:
        log.error(f"❌ Failed to write report to {a.out}: {e}")
        return 1

    # Telegram Notifications
    status_msg = f"🚀 *NSE Deal Report Generated*\n📅 Period: {start} to {end}\n📊 Symbols Scanned: {len(summary)}\n🤖 AI Analysed: {len(reports)}"
    send_telegram_alert(status_msg)
    
    doc_caption = f"NSE Deal Research Report ({start} to {end})\nTotal Deals: {len(summary)} | AI Analysed: {len(reports)}"
    send_telegram_document(a.out, doc_caption)
    
    # Optional PDF Generation
    if os.getenv("ENABLE_PDF_EXPORT", "").lower() == "true":
        pdf_path = a.out.rsplit(".", 1)[0] + ".pdf"
        if generate_pdf_report(a.out, pdf_path):
            send_telegram_document(pdf_path, f"PDF Version: {doc_caption}")

    # Console Summary
    print(f"\nWrote {a.out}: {len(summary)} symbol(s) with deals, {len(reports)} analysed.")
    for r in reports:
        pl = r.plan
        lvl = f" entry {pl.entry:,.2f} SL {pl.stop_loss:,.2f} T1 {pl.target1:,.2f}" if pl.entry else ""
        print(f"  {r.symbol:<12} {r.final_label:<20} {pl.direction}{lvl}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
