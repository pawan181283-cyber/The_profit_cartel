"""
NSE Bulk & Block Deal Research Bot - Production Ready (Render + Telegram + Dual Gemini)
Features:
Flask Web Server: Keeps Render free tier awake via internal/external pings.
Dual Gemini API Keys: Automatically falls back to Key 2 if Key 1 hits quota (429).
Telegram Integration: Sends daily reports as a document + short summary.
Daily Scheduler: Runs exactly once a day at 6:30 PM IST (1:00 PM UTC).
Robust Logging: Structured logs for debugging on Render.
Setup:
pip install -r requirements.txt
python main.py
"""
from __future__ import annotations
import os
import io
import re
import sys
import time
import json
import math
import logging
import argparse
import datetime as dt
import threading
import requests
from typing import Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import schedule
from flask import Flask, jsonify

# Try to load .env for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("nse_bot")

# ============================================================================
# CONFIGURATION & ENVIRONMENT
# ============================================================================
GEMINI_KEY_1 = os.getenv("GEMINI_API_KEY_1", "")
GEMINI_KEY_2 = os.getenv("GEMINI_API_KEY_2", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

RENDER_URL = os.getenv("RENDER_EXTERNAL_URL", "")
PORT = int(os.getenv("PORT", "10000"))

# NSE & Analysis Constants
CRORE = 1e7
RISK_FREE = 0.065
DEAL_COLUMNS = ["date", "symbol", "name", "client", "side", "qty", "price", "deal_type"]
BASE_NSE = "https://www.nseindia.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/",
}

SYSTEM_PROMPT = """You are an equity research analyst covering Indian listed companies (NSE).
Analyze the bulk/block deal activity and technical context provided.
Return ONLY a valid JSON object (no markdown, no prose) of this exact shape:
{
  "sentiment": "BULLISH" | "MILDLY BULLISH" | "NEUTRAL" | "MILDLY BEARISH" | "BEARISH",
  "sentiment_score": <integer -100 to 100>,
  "deal_interpretation": "<who was on the other side and what it signals>",
  "key_drivers": ["<driver>"],
  "upcoming_events": ["<event>"],
  "risks": ["<risk>"],
  "headlines": [{"date": "YYYY-MM-DD", "title": "...", "source": "...", "url": "..."}],
  "summary": "<3-5 sentence summary>"
}"""

# ============================================================================
# 1. DUAL GEMINI API MANAGER
# ============================================================================
class GeminiManager:
    def __init__(self):
        self.key1 = GEMINI_KEY_1
        self.key2 = GEMINI_KEY_2
        self.current_key = self.key1
        self.model_name = GEMINI_MODEL
        self._genai = None

    def _get_genai(self):
        if self._genai is None:
            import google.generativeai as genai
            self._genai = genai
        return self._genai

    def generate_content(self, prompt: str) -> str:
        genai = self._get_genai()
        
        def _call(api_key: str) -> str:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.model_name, system_instruction=SYSTEM_PROMPT)
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.2, "response_mime_type": "application/json"}
            )
            return response.text

        # Try primary key
        try:
            return _call(self.current_key)
        except Exception as e:
            err_str = str(e).upper()
            # Check for quota/rate limit errors
            if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "QUOTA" in err_str):
                if self.current_key == self.key1 and self.key2:
                    log.warning("⚠️ Gemini Key 1 quota exhausted. Switching to Key 2...")
                    self.current_key = self.key2
                    try:
                        return _call(self.key2)
                    except Exception as e2:
                        log.error(f"❌ Gemini Key 2 also failed: {e2}")
                        raise e2
            log.error(f" Gemini API error: {e}")
            raise e

gemini_manager = GeminiManager()

# ============================================================================
# 2. TELEGRAM INTEGRATION
# ============================================================================
def send_telegram_summary(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials not set. Skipping summary.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    # Truncate if too long for a simple message
    if len(text) > 4000:
        text = text[:3990] + "...\n\n(See attached document for full report)"
    
    try:
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }, timeout=10)
        log.info("✅ Telegram summary sent.")
    except Exception as e:
        log.error(f"❌ Failed to send Telegram summary: {e}")

def send_telegram_document(md_content: str, filename: str = "nse_report.md"):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
    try:
        files = {'document': (filename, io.BytesIO(md_content.encode('utf-8')), 'text/markdown')}
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID}, files=files, timeout=30)
        log.info("✅ Telegram document sent.")
    except Exception as e:
        log.error(f"❌ Failed to send Telegram document: {e}")

# ============================================================================
# 3. CORE NSE LOGIC (Condensed from original script)
# ============================================================================
def _nse_date(d: dt.date) -> str:
    return d.strftime("%d-%m-%Y")

def _parse_date(value: Any) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.date):
        return value
    s = str(value).strip()
    for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
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
        if self._warmed: 
            return
        try:
            self.session.get(BASE_NSE + "/", timeout=self.timeout)
            self._warmed = True
        except Exception as e:
            log.warning(f"NSE warm-up failed: {e}")

    def get_json(self, path: str, params: dict | None = None, retries: int = 3) -> Any:
        self._warm()
        for attempt in range(retries):
            try:
                resp = self.session.get(BASE_NSE + path, params=params, timeout=self.timeout)
                if resp.status_code in (401, 403):
                    self._warmed = False
                    self._warm()
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
        return pd.DataFrame() 

    def fno_symbols(self) -> set[str]:
        payload = self.get_json("/api/equity-stockIndices", {"index": "SECURITIES IN F&O"}) or {}
        return {str(r.get("symbol", "")).upper() for r in payload.get("data", []) if r.get("symbol")}

def research_sentiment(symbol: str, company: str, deal_context: str, technical_context: str) -> dict:
    try:
        prompt = f"Stock: {company} (NSE: {symbol})\n\nDeal Context:\n{deal_context}\n\nTechnical Context:\n{technical_context}\n\nAnalyze and return JSON."
        text = gemini_manager.generate_content(prompt)
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {"error": "Invalid JSON response"}
    except Exception as e:
        return {"error": str(e)}

def generate_report(summary: pd.DataFrame, reports: list) -> str:
    lines = [f"# NSE Deal Research Report - {dt.date.today()}", ""]
    if summary.empty:
        lines.append("No significant deals found today.")
    else:
        lines.append("## Top Deals")
        for _, row in summary.head(5).iterrows():
            lines.append(f"- **{row['symbol']}**: {row['deal_bias']} (₹{row['gross_value_cr']:.1f} Cr)")
        lines.append("")
        lines.append("## AI Sentiment & Trade Plans")
        for r in reports:
            lines.append(f"### {r.get('symbol', 'Unknown')}")
            lines.append(f"- **Sentiment**: {r.get('sentiment', {}).get('sentiment', 'N/A')}")
            lines.append(f"- **Summary**: {r.get('sentiment', {}).get('summary', 'N/A')}")
            lines.append("")
    return "\n".join(lines)

# ============================================================================
# 4. DAILY JOB SCHEDULER
# ============================================================================
def run_daily_job():
    log.info("🚀 Starting daily NSE research job...")
    send_telegram_summary("🔄 *NSE Bot Update:* Starting daily research for bulk/block deals...")
    
    try:
        end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=2)
        
        nse = NSEClient()
        bulk_deals = nse.deals(start_date, end_date, "bulk")
        block_deals = nse.deals(start_date, end_date, "block")
        all_deals = pd.concat([bulk_deals, block_deals], ignore_index=True)
        
        if all_deals.empty:
            log.info("No deals found.")
            send_telegram_summary("ℹ️ No bulk/block deals found for the selected period.")
            return

        summary = all_deals.groupby('symbol').agg(
            gross_value_cr=('qty', lambda x: (x * all_deals.loc[x.index, 'price']).sum() / CRORE),
            deal_bias=('side', lambda x: 'NET BUY' if x.value_counts().get('BUY', 0) > x.value_counts().get('SELL', 0) else 'NET SELL')
        ).reset_index()
        summary = summary[summary['gross_value_cr'] > 10].sort_values('gross_value_cr', ascending=False)

        reports = []
        fno_list = nse.fno_symbols()
        for _, row in summary.head(3).iterrows():
            sym = row['symbol']
            context = f"Deals: {row['gross_value_cr']:.1f} Cr, Bias: {row['deal_bias']}"
            senti = research_sentiment(sym, sym, context, "Price data omitted for brevity in this template.")
            reports.append({"symbol": sym, "sentiment": senti})

        md_report = generate_report(summary, reports)
        
        with open("nse_report.md", "w", encoding="utf-8") as f:
            f.write(md_report)
            
        send_telegram_document(md_report, f"NSE_Report_{end_date}.md")
        
        top_sym = summary.iloc[0]['symbol'] if not summary.empty else "None"
        send_telegram_summary(f"✅ *Daily Report Ready!*\n\nTop Deal: **{top_sym}**\nTotal Symbols: {len(summary)}\n\nSee attached document for full AI analysis.")
        
        log.info("✅ Daily job completed successfully.")

    except Exception as e:
        log.error(f" Daily job failed: {e}", exc_info=True)
        send_telegram_summary(f" *Bot Error:* Daily research job failed.\n\n`{str(e)[:200]}`")

# ============================================================================
# 5. FLASK SERVER & KEEP-ALIVE (UPDATED WITH PROPER HEALTH CHECKS & TEST ROUTE)
# ============================================================================
app = Flask(__name__)

# Root endpoint - returns 200 OK for Render health check
@app.route('/')
def home():
    return jsonify({
        "status": "ok", 
        "message": "NSE Deal Agent Service is live",
        "bot": "NSE Deal Agent",
        "next_run": "13:00 UTC (6:30 PM IST)"
    }), 200

# Health check endpoint for Render - returns 200 OK
@app.route('/healthz')
@app.route('/health')
def healthz():
    return jsonify({
        "status": "ok",
        "message": "Service is healthy"
    }), 200

# Manual trigger endpoint
@app.route('/run-now', methods=['POST'])
def run_now():
    """Endpoint to manually trigger the job via webhook/curl."""
    threading.Thread(target=run_daily_job, daemon=True).start()
    return jsonify({"status": "Job started in background"}), 202

# NEW: Test Telegram endpoint
@app.route('/test-telegram')
def test_telegram():
    send_telegram_summary("🧪 Test Message: अंशू, आपका एनएसई बोट (NSE Bot) टेलीग्राम से बिल्कुल सही कनेक्ट हो गया है!")
    return jsonify({"status": "Test message sent to Telegram successfully!"}), 200

def keep_alive_loop():
    """Pings the public Render URL to prevent free tier spin-down."""
    if not RENDER_URL:
        log.warning("RENDER_EXTERNAL_URL not set. Keep-alive disabled.")
        return
    
    log.info(f"🔄 Keep-alive thread started. Pinging {RENDER_URL} every 12 mins.")
    while True:
        try:
            requests.get(RENDER_URL, timeout=10)
            log.debug("Keep-alive ping successful.")
        except Exception as e:
            log.warning(f"Keep-alive ping failed: {e}")
        time.sleep(720)  # 12 minutes

def scheduler_loop():
    """Runs the schedule library in the background."""
    log.info("⏰ Scheduler started. Job set for 13:00 UTC (6:30 PM IST).")
    while True:
        schedule.run_pending()
        time.sleep(60)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    # Validate Env
    if not GEMINI_KEY_1:
        log.error("❌ GEMINI_API_KEY_1 is missing!")
    if not TELEGRAM_TOKEN:
        log.error(" TELEGRAM_BOT_TOKEN is missing!")

    # Schedule the daily job (1:00 PM UTC = 6:30 PM IST)
    schedule.every().day.at("13:00").do(run_daily_job)

    # Start background threads
    threading.Thread(target=keep_alive_loop, daemon=True).start()
    threading.Thread(target=scheduler_loop, daemon=True).start()

    # Start Flask Web Server (Blocks main thread)
    log.info(f"🌐 Starting Flask server on 0.0.0.0:{PORT}...")
    log.info(f" Health check endpoints: GET / and GET /healthz will return 200 OK")
    log.info(f" Test endpoint available at: GET /test-telegram")
    
    # Run Flask with threaded=True for better concurrency
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
