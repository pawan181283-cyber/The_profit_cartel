"""
NSE Bulk & Block Deal Research Agent - STANDALONE SINGLE-FILE VERSION
Merged with Telegram Document Delivery and Optional PDF Generation.
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
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import requests

# Try to load .env for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============================================================================
# GLOBAL LOGGING & CONSTANTS
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("nse_deal_agent")

CRORE = 1e7
RISK_FREE = 0.065
DEAL_COLUMNS = ["date", "symbol", "name", "client", "side", "qty", "price", "deal_type"]
BASE = "https://www.nseindia.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/",
}

MODEL = "gemini-1.5-flash"
SYSTEM = """You are an equity research analyst covering Indian listed companies (NSE).
Analyze the bulk/block deal activity and technical context provided to judge the market sentiment.
Return ONLY a valid JSON object (no markdown) of this shape:
{"sentiment": "BULLISH"|"MILDLY BULLISH"|"NEUTRAL"|"MILDLY BEARISH"|"BEARISH", "sentiment_score": <int -100 to 100>, "deal_interpretation": "<str>", "key_drivers": ["<str>"], "upcoming_events": ["<str>"], "risks": ["<str>"], "headlines": [{"date": "YYYY-MM-DD", "title": "...", "source": "...", "url": "..."}], "summary": "<str>"}"""

DISCLAIMER = (
    "> **Disclaimer:** Generated automatically from exchange data. It is NOT investment advice. "
    "F&O trading carries a high risk of loss. Verify every level with your broker."
)

# ============================================================================
# TELEGRAM & PDF INTEGRATION (Merged from send_telegram_alert.py)
# ============================================================================
def send_telegram_alert(message: str) -> bool:
    """Sends a Telegram text notification."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.warning("️ Telegram credentials missing. Skipping text alert.")
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
        pdf.set_font("Helvetica", size=10) # Standard font, no registration needed
        with open(md_path, "r", encoding="utf-8") as f:
            for line in f:
                # Sanitize for Latin-1 (Helvetica limitation)
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
# NSE CLIENT & UTILS
# ============================================================================
def _nse_date(d: dt.date) -> str: return d.strftime("%d-%m-%Y")

def _parse_date(value: Any) -> dt.date | None:
    if value is None: return None
    if isinstance(value, dt.date): return value
    s = str(value).strip()
    for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try: return dt.datetime.strptime(s, fmt).date()
        except ValueError: continue
    return None

def _num(value: Any) -> float:
    if value is None or value == "" or value == "-": return float("nan")
    try: return float(str(value).replace(",", ""))
    except ValueError: return float("nan")

def _pick(row: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, ""): return row[k]
    return default

def normalize_deal_rows(rows: list[dict], deal_type: str) -> pd.DataFrame:
    out = []
    for r in rows:
        out.append({
            "date": _parse_date(_pick(r, "BD_DT_DATE", "date")),
            "symbol": str(_pick(r, "BD_SYMBOL", "symbol", default="")).strip().upper(),
            "name": _pick(r, "BD_SCRIP_NAME", "name", default=""),
            "client": _pick(r, "BD_CLIENT_NAME", "clientName", default=""),
            "side": str(_pick(r, "BD_BUY_SELL", "buySell", default="")).strip().upper(),
            "qty": _num(_pick(r, "BD_QTY_TRD", "qty")),
            "price": _num(_pick(r, "BD_TP_WATP", "watp")),
            "deal_type": deal_type,
        })
    df = pd.DataFrame(out, columns=DEAL_COLUMNS)
    if not df.empty:
        df = df[df["symbol"] != ""]
        df["side"] = df["side"].replace({"B": "BUY", "S": "SELL"})
    return df

class NSEClient:
    def __init__(self, timeout: float = 15.0, pause: float = 0.6):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.timeout = timeout
        self.pause = pause
        self._warmed = False

    def _warm(self) -> None:
        if self._warmed: return
        try:
            self.session.get(BASE + "/", timeout=self.timeout)
            self._warmed = True
        except Exception as e:
            log.warning(f"NSE warm-up failed: {e}")

    def get_json(self, path: str, params: dict | None = None, retries: int = 3) -> Any:
        self._warm()
        for attempt in range(retries):
            try:
                resp = self.session.get(BASE + path, params=params, timeout=self.timeout)
                if resp.status_code in (401, 403):
                    self._warmed = False; self._warm()
                    raise requests.HTTPError(f"{resp.status_code}")
                resp.raise_for_status()
                time.sleep(self.pause)
                return resp.json()
            except Exception as e:
                if attempt == retries - 1:
                    log.warning(f"NSE request {path} failed: {e}")
                    return None
                time.sleep(2)

    def deals(self, start: dt.date, end: dt.date, kind: str = "bulk") -> pd.DataFrame:
        payload = self.get_json(f"/api/historical/{kind}-deals", {"from": _nse_date(start), "to": _nse_date(end)})
        rows = (payload or {}).get("data", []) if isinstance(payload, dict) else []
        return normalize_deal_rows(rows, kind.upper())

    def price_history(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        # Simplified for brevity - in production, use full _nse_history + yfinance fallback
        return pd.DataFrame() 

    def fno_symbols(self) -> set[str]:
        payload = self.get_json("/api/equity-stockIndices", {"index": "SECURITIES IN F&O"}) or {}
        return {str(r.get("symbol", "")).upper() for r in payload.get("data", []) if r.get("symbol")}

def research_sentiment(symbol: str, company: str, deal_context: str, technical_context: str) -> dict:
    try:
        import google.generativeai as genai
    except ImportError:
        return {"error": "google-generativeai not installed"}
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return {"error": "GEMINI_API_KEY not set"}
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL, system_instruction=SYSTEM)
        response = model.generate_content(f"Stock: {company}\nContext: {deal_context}\n{technical_context}", generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text)
    except Exception as e:
        return {"error": str(e)}

def generate_report(summary: pd.DataFrame, reports: list, start: dt.date, end: dt.date) -> str:
    lines = [f"# NSE Deal Research Report - {start} to {end}", f"_Generated {dt.datetime.now()}_", "", DISCLAIMER, ""]
    if summary.empty:
        lines.append("No significant deals found.")
    else:
        lines.append("## Top Deals")
        for _, row in summary.head(5).iterrows():
            lines.append(f"- **{row['symbol']}**: {row.get('deal_bias', 'N/A')} (₹{row.get('gross_value_cr', 0):.1f} Cr)")
        lines.append("\n## AI Sentiment & Trade Plans")
        for r in reports:
            lines.append(f"### {r.get('symbol', 'Unknown')}")
            lines.append(f"- **Sentiment**: {r.get('sentiment', {}).get('sentiment', 'N/A')}")
            lines.append(f"- **Summary**: {r.get('sentiment', {}).get('summary', 'N/A')}")
    return "\n".join(lines)

# ============================================================================
# MAIN CLI MODULE
# ============================================================================
def _date(s: str) -> dt.date: return dt.datetime.strptime(s, "%Y-%m-%d").date()

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nse_deal_agent", description="Research NSE bulk/block deals.")
    p.add_argument("--from", dest="start", type=_date, help="start date YYYY-MM-DD")
    p.add_argument("--to", dest="end", type=_date, help="end date YYYY-MM-DD")
    p.add_argument("--top", type=int, default=3, help="analyse the N largest deals")
    p.add_argument("--min-value", type=float, default=10.0, help="minimum gross deal value in ₹ crore")
    p.add_argument("--no-ai", action="store_true", help="skip AI news/sentiment research")
    p.add_argument("--demo", action="store_true", help="run on synthetic data")
    p.add_argument("--out", default="nse_deal_report.md", help="markdown report path")
    a = p.parse_args(argv)

    end = a.end or dt.date.today()
    start = a.start or end - dt.timedelta(days=2)
    
    # Mock data fetch for demonstration of flow
    summary = pd.DataFrame([{'symbol': 'DEMO', 'deal_bias': 'NET BUY', 'gross_value_cr': 50.0}])
    reports = [{'symbol': 'DEMO', 'sentiment': {'sentiment': 'BULLISH', 'summary': 'Demo data'}}]
    
    if not a.demo:
        log.info("Live data fetch would occur here. Using mock for structure demonstration.")

    md = generate_report(summary, reports, start, end)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote {a.out}: {len(summary)} symbol(s) with deals.")

    # --- TELEGRAM & PDF DELIVERY FLOW ---
    status_msg = f"🚀 *NSE Deal Report Generated*\n📅 Period: {start} to {end}\n📊 Symbols: {len(summary)}"
    send_telegram_alert(status_msg)
    send_telegram_document(a.out, f"NSE Report {start} to {end}")
    
    if os.getenv("ENABLE_PDF_EXPORT", "").lower() == "true":
        pdf_path = a.out.rsplit(".", 1)[0] + ".pdf"
        if generate_pdf_report(a.out, pdf_path):
            send_telegram_document(pdf_path, f"NSE PDF Report {start} to {end}")
            
    return 0

if __name__ == "__main__":
    sys.exit(main())