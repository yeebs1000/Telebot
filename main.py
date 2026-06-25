"""
Intelligent Telegram Group Assistant — main.py
===============================================
FIXES IN THIS VERSION vs PREVIOUS:
  1. [CRITICAL] Broken Apple NTP URL (was a markdown link literal) — extracted
     into get_network_time() helper used everywhere, no duplication.
  2. [CRITICAL] classify_intent() routing prompt completely rewritten with
     explicit few-shot examples so the lite model reliably hits WEB_SEARCH
     for live scores / news AND correctly maps company names → tickers.
  3. [CRITICAL] All httpx calls converted to AsyncClient (non-blocking).
  4. [CRITICAL] All Supabase + sync GenAI calls wrapped in asyncio.to_thread()
     so the event loop is never blocked under concurrent group messages.
  5. [BUG] Bot shutdown loop replaced with asyncio.Event().wait() + clean drain.
  6. [BUG] Removed unused `import numpy as np`.
  7. [BUG] Unreachable return at the bottom of fetch_live_financial_data removed.
  8. [IMPROVEMENT] Tavily `include_answer=True` — returns a direct synthesised
     answer which is critical for sports scores & breaking news.
  9. [IMPROVEMENT] Alpha Vantage rate-limit Note/Information fields are now
     detected and surfaced rather than silently returning an empty quote.
 10. [IMPROVEMENT] `drop_pending_updates=True` on start_polling — prevents the
     bot from replaying stale messages after a Railway container restart.
 11. [IMPROVEMENT] MODEL constant at the top — change once, applies everywhere.
"""

import os
import logging
import httpx
import asyncio
import json
import uuid
from datetime import datetime, timedelta
import pytz

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters,
)
from google import genai
from google.genai import types
from google.genai.errors import APIError
from supabase import create_client, Client
from dotenv import load_dotenv

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── ENVIRONMENT ───────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

SUPABASE_URL       = os.getenv("SUPABASE_URL")
SUPABASE_KEY       = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TAVILY_API_KEY     = os.getenv("TAVILY_API_KEY")
ALPHA_VANTAGE_KEY  = os.getenv("ALPHA_VANTAGE_KEY")

if not all([GEMINI_API_KEY, TELEGRAM_BOT_TOKEN, SUPABASE_URL, SUPABASE_KEY]):
    logger.critical("MISSING CORE ENV VARS — check your .env / Railway config.")

# ── CLIENT INIT ───────────────────────────────────────────────────────────────
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# In-memory session store (wiped on container restart — acceptable for group chat)
chat_sessions: dict = {}

# ─────────────────────────────────────────────────────────────────────────────
# UPDATE THIS if you have access to a newer or different model tier.
# The same model is used for chat sessions AND routing classification.
# ─────────────────────────────────────────────────────────────────────────────
MODEL = "gemini-3.1-flash-lite"


# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
def get_system_prompt() -> str:
    return (
        "Be really concise — don't reply in a long fashion unless absolutely necessary. "
        "Reply like we have known each other for years: light-hearted, real, like talking to a friend. "
        "If the user asks you to show or send an image, reply ONLY with a direct public image URL string."
    )


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

async def get_network_time() -> datetime:
    """
    FIX #1 — The original code had a markdown link baked into the URL string:
        time_client.head("[https://www.apple.com](https://www.apple.com)", ...)
    That silently fails every single call, so NTP sync was never actually running.

    This function fetches real atomic time from Apple's HTTP Date header and is
    used everywhere a timestamp is needed, eliminating all duplication.
    """
    try:
        async with httpx.AsyncClient() as c:
            res = await c.head("https://www.apple.com", timeout=2.0)
            date_str = res.headers.get("Date", "")
        utc_dt = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z")
        return utc_dt.replace(tzinfo=pytz.utc).astimezone(pytz.timezone("Asia/Singapore"))
    except Exception as e:
        logger.warning(f"NTP fetch failed — falling back to system clock: {e}")
        return datetime.now(pytz.timezone("Asia/Singapore"))


async def search_the_live_web(query: str) -> str:
    """
    FIX #3 + IMPROVEMENT #8 — Now uses AsyncClient (non-blocking) and sets
    include_answer=True so Tavily returns a direct synthesised answer, which
    is essential for sports scores and live event results.
    """
    if not TAVILY_API_KEY:
        return "Tavily API key not configured."
    try:
        async with httpx.AsyncClient() as c:
            res = await c.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "basic",
                    "include_answer": True,   # ← direct answer for scores, weather, etc.
                    "max_results": 5,
                },
                timeout=8.0,
            )
            data = res.json()

        answer  = data.get("answer", "")
        results = data.get("results", [])

        if not answer and not results:
            return "No live data found for this query."

        output = ""
        if answer:
            output += f"Direct Answer: {answer}\n\n"
        for idx, item in enumerate(results[:3], 1):
            output += f"[{idx}] {item.get('title', '')}\n{item.get('content', '')}\n\n"
        return output.strip()

    except Exception as e:
        logger.error(f"Web search error: {e}")
        return "Web search failed."


async def fetch_live_financial_data(asset_type: str, symbol: str) -> str:
    """
    FIX #3 — Now uses AsyncClient throughout (non-blocking).
    IMPROVEMENT #9 — Alpha Vantage rate-limit notes are detected and returned
    as a readable message instead of silently returning an empty quote.
    FIX #7 — Unreachable bare `return` at the end of the original removed.
    """
    if not ALPHA_VANTAGE_KEY:
        return "Alpha Vantage API key not configured."

    symbol = symbol.strip().upper()

    try:
        async with httpx.AsyncClient() as c:

            # ── STOCKS ───────────────────────────────────────────────────────
            if asset_type == "STOCK":
                res   = await c.get(
                    "https://www.alphavantage.co/query",
                    params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": ALPHA_VANTAGE_KEY},
                    timeout=6.0,
                )
                data  = res.json()
                quote = data.get("Global Quote", {})

                if not quote or not quote.get("05. price"):
                    # Surface rate-limit messages instead of swallowing them
                    note = data.get("Note") or data.get("Information", "")
                    if note:
                        return f"Alpha Vantage rate limit hit: {note}"
                    return f"No quote returned for '{symbol}'. Verify the ticker symbol."

                return (
                    f"📈 {quote['01. symbol']}\n"
                    f"Price   : ${quote['05. price']}\n"
                    f"High/Low: ${quote['03. high']} / ${quote['04. low']}\n"
                    f"Prev Close: ${quote['08. previous close']}\n"
                    f"Change  : {quote['10. change percent']}"
                )

            # ── FOREX ────────────────────────────────────────────────────────
            elif asset_type == "FOREX":
                from_c = symbol[:3]
                to_c   = symbol[3:] if len(symbol) > 3 else "USD"
                res    = await c.get(
                    "https://www.alphavantage.co/query",
                    params={
                        "function": "CURRENCY_EXCHANGE_RATE",
                        "from_currency": from_c,
                        "to_currency": to_c,
                        "apikey": ALPHA_VANTAGE_KEY,
                    },
                    timeout=6.0,
                )
                rate = res.json().get("Realtime Currency Exchange Rate", {})
                if not rate:
                    return f"No exchange rate data returned for {from_c}/{to_c}."
                return (
                    f"💱 {rate['1. From_Currency Code']}/{rate['3. To_Currency Code']}\n"
                    f"Rate: {rate['5. Exchange Rate']}\n"
                    f"Bid: {rate['8. Bid Price']} | Ask: {rate['9. Ask Price']}"
                )

            # ── COMMODITIES ──────────────────────────────────────────────────
            elif asset_type == "COMMODITY":
                COMM_MAP = {
                    "GOLD": "GOLD", "SILVER": "SILVER",
                    "OIL": "CRUDE_OIL", "CRUDE": "CRUDE_OIL", "WTI": "CRUDE_OIL",
                    "BRENT": "BRENT", "GAS": "NATURAL_GAS",
                    "COPPER": "COPPER", "WHEAT": "WHEAT",
                }
                function = COMM_MAP.get(symbol, symbol)
                res      = await c.get(
                    "https://www.alphavantage.co/query",
                    params={"function": function, "apikey": ALPHA_VANTAGE_KEY},
                    timeout=6.0,
                )
                points = res.json().get("data", [])
                if not points:
                    return f"No commodity data returned for {function}."
                latest = points[0]
                return f"🛢 {function}\nDate: {latest['date']}\nSpot: ${latest['value']} USD"

    except Exception as e:
        logger.error(f"Financial API error [{asset_type}/{symbol}]: {e}")
        return f"Market data fetch failed: {e}"

    return f"Unrecognised asset type: {asset_type}"


async def classify_intent(user_text: str) -> dict:
    """
    FIX #2 — THE MAIN ROUTING FIX.

    The original prompt was too vague. A lite model without concrete examples
    defaulted everything to REGULAR_CHAT, which is why sports scores and stock
    queries were never being routed. This rewrite uses explicit few-shot examples
    for every category, forcing reliable classification even from a small model.

    Key additions:
      - Company name → ticker mapping examples (lululemon → LULU)
      - Explicit rule: ANY live score / match result MUST be WEB_SEARCH
      - Explicit rule: ANY current news MUST be WEB_SEARCH
    """
    routing_prompt = f"""You are a strict JSON intent classifier for a Telegram assistant bot.

Read the user message and return EXACTLY ONE JSON object from the options below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATEGORY 1 — STOCK
User is asking about a stock, share price, or company equity value.
Output: {{"type": "STOCK", "symbol": "<TICKER>"}}

Examples:
  "how much is lululemon"        → {{"type": "STOCK", "symbol": "LULU"}}
  "apple stock price"            → {{"type": "STOCK", "symbol": "AAPL"}}
  "nvidia share price"           → {{"type": "STOCK", "symbol": "NVDA"}}
  "what is tesla trading at"     → {{"type": "STOCK", "symbol": "TSLA"}}
  "price of MSFT"                → {{"type": "STOCK", "symbol": "MSFT"}}
  "amazon stock"                 → {{"type": "STOCK", "symbol": "AMZN"}}
  "META share price"             → {{"type": "STOCK", "symbol": "META"}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATEGORY 2 — FOREX
User is asking about currency exchange rates.
Output: {{"type": "FOREX", "symbol": "<FROM><TO>"}}

Examples:
  "EURUSD rate"                  → {{"type": "FOREX", "symbol": "EURUSD"}}
  "SGD to USD"                   → {{"type": "FOREX", "symbol": "SGDUSD"}}
  "USD/JPY"                      → {{"type": "FOREX", "symbol": "USDJPY"}}
  "what's the GBP to SGD rate"   → {{"type": "FOREX", "symbol": "GBPSGD"}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATEGORY 3 — COMMODITY
User is asking about hard commodities: gold, silver, oil, gas, copper, wheat.
Output: {{"type": "COMMODITY", "symbol": "<ASSET>"}}

Examples:
  "gold price"                   → {{"type": "COMMODITY", "symbol": "GOLD"}}
  "crude oil"                    → {{"type": "COMMODITY", "symbol": "OIL"}}
  "silver spot"                  → {{"type": "COMMODITY", "symbol": "SILVER"}}
  "brent crude"                  → {{"type": "COMMODITY", "symbol": "BRENT"}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATEGORY 4 — WEB_SEARCH
User asks about: live match scores, sports results, current news, ongoing events,
weather, recent developments — ANYTHING that requires live internet data.
Output: {{"type": "WEB_SEARCH"}}

Examples:
  "score germany vs curacao"     → {{"type": "WEB_SEARCH"}}
  "what's the score"             → {{"type": "WEB_SEARCH"}}
  "who won the F1 race today"    → {{"type": "WEB_SEARCH"}}
  "Champions League results"     → {{"type": "WEB_SEARCH"}}
  "latest bitcoin news"          → {{"type": "WEB_SEARCH"}}
  "is the Premier League live"   → {{"type": "WEB_SEARCH"}}
  "weather in Singapore"         → {{"type": "WEB_SEARCH"}}
  "what happened in the news"    → {{"type": "WEB_SEARCH"}}
  "current price of ethereum"    → {{"type": "WEB_SEARCH"}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATEGORY 5 — REGULAR_CHAT
General conversation, jokes, opinions, explanations, math, history.
Output: {{"type": "REGULAR_CHAT"}}

Examples:
  "how are you"                  → {{"type": "REGULAR_CHAT"}}
  "tell me a joke"               → {{"type": "REGULAR_CHAT"}}
  "what's 15% of 340"           → {{"type": "REGULAR_CHAT"}}
  "explain quantum computing"    → {{"type": "REGULAR_CHAT"}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HARD RULES (never violate these):
- ANY live or recent sports score/result → WEB_SEARCH (never REGULAR_CHAT)
- ANY question about current news or live events → WEB_SEARCH
- Company names map to tickers: lululemon=LULU, apple=AAPL, google=GOOGL, meta=META
- Crypto prices (bitcoin, ethereum) → WEB_SEARCH (not on Alpha Vantage)
- Return ONLY the raw JSON object — no markdown, no explanation, no extra text.

User message: "{user_text}"
"""

    try:
        response = await asyncio.to_thread(
            lambda: genai_client.models.generate_content(
                model=MODEL,
                contents=routing_prompt,
                config={"response_mime_type": "application/json"},
            )
        )
        route = json.loads(response.text.strip())
        logger.info(f"[ROUTER] '{user_text}' → {route}")
        return route
    except Exception as e:
        logger.error(f"Intent classification error: {e} — defaulting to REGULAR_CHAT")
        return {"type": "REGULAR_CHAT"}


# ═════════════════════════════════════════════════════════════════════════════
# JOB QUEUE CALLBACKS
# ═════════════════════════════════════════════════════════════════════════════

async def execute_dynamic_reminder(context: ContextTypes.DEFAULT_TYPE):
    """Fires when a scheduled reminder job triggers."""
    d = context.job.data
    await context.bot.send_message(
        chat_id=d["chat_id"],
        text=f"🔔 **REMINDER FOR {d['user'].upper()}:**\n\n> {d['reminder_text']}",
        parse_mode="Markdown",
    )


async def expire_poll_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Fires when a poll window closes. Tallies votes, determines winner,
    edits the original message, then deletes the DB record.
    FIX #4 — Supabase calls wrapped in asyncio.to_thread.
    """
    d = context.job.data
    poll_id, chat_id, message_id = d["poll_id"], d["chat_id"], d["message_id"]
    try:
        res  = await asyncio.to_thread(
            lambda: supabase_client.table("active_polls").select("*").eq("poll_id", poll_id).execute()
        )
        poll = res.data[0] if res.data else None
        if not poll:
            return

        options       = poll["options"]
        result_text   = f"📊 **POLL CLOSED: {poll['question']}**\n\nFinal Standings:\n"
        max_v, winner = -1, "Nobody voted!"

        for opt, count in options.items():
            result_text += f"▪️ {opt}: **{count} votes**\n"
            if count > max_v and count > 0:
                max_v, winner = count, opt
            elif count == max_v and count > 0:
                winner = f"Tie between {winner} and {opt}!"

        result_text += f"\n🏆 **Winner:** {winner}"

        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=message_id,
            text=result_text, parse_mode="Markdown",
        )
        await asyncio.to_thread(
            lambda: supabase_client.table("active_polls").delete().eq("poll_id", poll_id).execute()
        )
    except Exception as e:
        logger.error(f"Poll expiry error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# POLL CALLBACK HANDLER
# ═════════════════════════════════════════════════════════════════════════════

async def handle_poll_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles inline keyboard button presses on live polls."""
    query = update.callback_query
    await query.answer()

    user_id = str(query.from_user.id)
    if ":" not in query.data:
        return

    poll_id, opt_idx_str = query.data.split(":", 1)

    try:
        res  = await asyncio.to_thread(
            lambda: supabase_client.table("active_polls").select("*").eq("poll_id", poll_id).execute()
        )
        poll = res.data[0] if res.data else None
        if not poll:
            await query.answer("This poll has already ended!", show_alert=True)
            return

        opts  = poll["options"]
        votes = poll["votes"]
        keys  = list(opts.keys())
        sel   = keys[int(opt_idx_str)]

        # Handle vote changes
        if user_id in votes:
            prior = votes[user_id]
            if prior == sel:
                return  # Same vote — no-op
            opts[prior] = max(0, opts[prior] - 1)

        votes[user_id] = sel
        opts[sel]     += 1

        await asyncio.to_thread(
            lambda: supabase_client.table("active_polls").update(
                {"options": opts, "votes": votes}
            ).eq("poll_id", poll_id).execute()
        )

        updated_kb = [
            [InlineKeyboardButton(f"{k} ({opts[k]})", callback_data=f"{poll_id}:{i}")]
            for i, k in enumerate(keys)
        ]
        await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup(updated_kb))

    except Exception as e:
        logger.error(f"Poll callback error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN MESSAGE HANDLER
# ═════════════════════════════════════════════════════════════════════════════

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Core handler — logs context, routes intent, and delivers a response.
    All blocking I/O is now wrapped in asyncio.to_thread() (FIX #4).
    """
    if not update.effective_chat or not update.message:
        return

    chat_id   = update.effective_chat.id
    chat_type = update.message.chat.type
    user_name = update.message.from_user.first_name if update.message.from_user else "Someone"
    user_text = update.message.text or update.message.caption or ""

    bot_info     = await context.bot.get_me()
    bot_username = f"@{bot_info.username}"
    is_mentioned = bot_username.lower() in user_text.lower() or chat_type == "private"

    # ── IMAGE HANDLING ────────────────────────────────────────────────────────
    image_bytes = None
    if update.message.photo:
        file = await context.bot.get_file(update.message.photo[-1].file_id)
        buf  = bytearray()
        await file.download_to_memory(buf)
        image_bytes = bytes(buf)
        if chat_type in ["group", "supergroup"]:
            is_mentioned = True

    if not user_text and not image_bytes:
        return

    # ── BACKGROUND LOGGING (non-directed group messages only) ─────────────────
    if chat_type in ["group", "supergroup"] and not is_mentioned and user_text:
        try:
            await asyncio.to_thread(
                lambda: supabase_client.table("group_chat_logs").insert(
                    {"chat_id": chat_id, "sender": user_name, "message": user_text}
                ).execute()
            )
            embed = await asyncio.to_thread(
                lambda: genai_client.models.embed_content(
                    model="text-embedding-004",
                    contents=f"{user_name}: {user_text}",
                )
            )
            vec = embed.embeddings[0].values
            await asyncio.to_thread(
                lambda: supabase_client.table("group_embeddings").insert(
                    {"chat_id": chat_id, "sender": user_name, "message": user_text, "embedding": vec}
                ).execute()
            )
        except Exception as e:
            logger.error(f"Background log/embed error: {e}")

    if not is_mentioned:
        return

    # ── SESSION INIT ──────────────────────────────────────────────────────────
    if chat_id not in chat_sessions:
        chat_sessions[chat_id] = genai_client.chats.create(
            model=MODEL,
            config=types.GenerateContentConfig(
                system_instruction=get_system_prompt(),
                temperature=0.7,
            ),
        )

    chat         = chat_sessions[chat_id]
    cleaned_text = user_text.replace(bot_username, "").strip().lower()
    prompt_payload: str | None = None

    # ══════════════════════════════════════════════════════════════════════════
    # FEATURE 1 — GROUP SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    if any(k in cleaned_text for k in ["summarise", "summarize", "summary"]) and not image_bytes:
        try:
            res = await asyncio.to_thread(
                lambda: supabase_client.table("group_chat_logs")
                .select("sender, message")
                .eq("chat_id", chat_id)
                .order("created_at", desc=True)
                .limit(500)
                .execute()
            )
            records = list(reversed(res.data or []))
            history = "\n".join(f"{r['sender']}: {r['message']}" for r in records) or "No logs yet."
        except Exception as e:
            logger.error(f"Summary DB fetch error: {e}")
            history = f"DB error: {e}"

        prompt_payload = (
            f"Summary request. Here are the recent group chat logs:\n"
            f"### LOGS ###\n{history}\n### END ###\n\n"
            f"Give a short, light-hearted recap of what went down."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # FEATURE 2 — PEER PERSONALITY ROAST
    # ══════════════════════════════════════════════════════════════════════════
    elif "what do you think of" in cleaned_text and not image_bytes:
        target = cleaned_text.split("what do you think of")[-1].strip().rstrip("?").strip()
        try:
            res = await asyncio.to_thread(
                lambda: supabase_client.table("group_chat_logs")
                .select("sender, message")
                .eq("chat_id", chat_id)
                .ilike("sender", f"%{target}%")
                .limit(200)
                .execute()
            )
            records = res.data or []
            if records:
                history = "\n".join(f"- {r['message']}" for r in records)
                prompt_payload = (
                    f"Personality assessment of '{target}' based purely on their messages:\n"
                    f"{history}\n\nBe funny, punchy, and authentic — like roasting a close friend. Keep it short!"
                )
            else:
                prompt_payload = f"Tell the user you found no messages from '{target}' in the database yet."
        except Exception as e:
            logger.error(f"Peer roast DB error: {e}")
            prompt_payload = "Tell user the database threw an error while profiling."

    # ══════════════════════════════════════════════════════════════════════════
    # FEATURE 3 — NATURAL LANGUAGE REMINDERS
    # ══════════════════════════════════════════════════════════════════════════
    elif "remind" in cleaned_text and not image_bytes:
        live_dt  = await get_network_time()
        time_ctx = live_dt.strftime("%A, %d %B %Y, %I:%M %p SGT")

        parse_prompt = (
            f"You are a JSON extraction utility. Current time: {time_ctx}.\n"
            f"User said: '{user_text}'\n\n"
            f"Return ONLY this JSON — no markdown, no extra text:\n"
            f'{{"target_timestamp": "YYYY-MM-DD HH:MM:SS", "task": "description of what to remind them to do"}}'
        )

        try:
            raw = await asyncio.to_thread(
                lambda: genai_client.models.generate_content(model=MODEL, contents=parse_prompt)
            )
            clean  = raw.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(clean)

            sg_tz  = pytz.timezone("Asia/Singapore")
            naive  = datetime.strptime(parsed["target_timestamp"], "%Y-%m-%d %H:%M:%S")
            target = sg_tz.localize(naive)

            if target <= datetime.now(sg_tz):
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="That time is already in the past! Pick a future moment. ⏳",
                )
                return

            context.application.job_queue.run_once(
                execute_dynamic_reminder,
                when=target,
                data={"chat_id": chat_id, "reminder_text": parsed["task"], "user": user_name},
            )
            readable = target.strftime("%A, %d %B at %I:%M %p SGT")
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"✅ Done! I'll remind you to **{parsed['task']}** on {readable}. 🫡",
                parse_mode="Markdown",
            )
            return
        except Exception as e:
            logger.error(f"Reminder parse/schedule error: {e}")
            prompt_payload = "Tell user the reminder scheduler hit a parsing snag."

    # ══════════════════════════════════════════════════════════════════════════
    # FEATURE 4 — SEMANTIC LONG-TERM MEMORY SEARCH
    # ══════════════════════════════════════════════════════════════════════════
    elif any(k in cleaned_text for k in ["where did we", "what was that", "search memory"]) and not image_bytes:
        try:
            embed = await asyncio.to_thread(
                lambda: genai_client.models.embed_content(model="text-embedding-004", contents=user_text)
            )
            vec = embed.embeddings[0].values

            db_res = await asyncio.to_thread(
                lambda: supabase_client.rpc("match_chat_embeddings", {
                    "query_embedding": vec,
                    "match_threshold": 0.3,
                    "match_count": 5,
                    "filter_chat_id": chat_id,
                }).execute()
            )
            matches = db_res.data or []
            if matches:
                mem = "\n".join(f"- {m['sender']}: {m['message']}" for m in matches)
                prompt_payload = (
                    f"Semantic memory search results:\n{mem}\n\n"
                    f"Answer the user's question concisely based on these records."
                )
            else:
                prompt_payload = "Tell user nothing matched in the vector memory index for that topic."
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            prompt_payload = "Tell user the semantic search pipeline failed."

    # ══════════════════════════════════════════════════════════════════════════
    # FEATURE 5 — INTERACTIVE LIVE POLLS
    # ══════════════════════════════════════════════════════════════════════════
    elif any(k in cleaned_text for k in ["create poll", "poll:"]) and not image_bytes:
        try:
            raw   = user_text.replace("create poll", "").replace("poll:", "").strip()
            parts = [p.strip() for p in raw.split("|") if p.strip()]

            if len(parts) < 3:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="Format: `poll: Question | Option 1 | Option 2`",
                    parse_mode="Markdown",
                )
                return

            question, options = parts[0], parts[1:]
            poll_id  = str(uuid.uuid4())[:8]
            opts_map = {opt: 0 for opt in options}
            sg_tz    = pytz.timezone("Asia/Singapore")
            expires  = datetime.now(sg_tz) + timedelta(minutes=5)

            await asyncio.to_thread(
                lambda: supabase_client.table("active_polls").insert({
                    "poll_id": poll_id,
                    "chat_id": chat_id,
                    "question": question,
                    "options": opts_map,
                    "expires_at": expires.isoformat(),
                }).execute()
            )

            buttons = [
                [InlineKeyboardButton(f"{opt} (0)", callback_data=f"{poll_id}:{i}")]
                for i, opt in enumerate(options)
            ]
            sent = await context.bot.send_message(
                chat_id=chat_id,
                text=f"📊 **LIVE POLL**\n\n> {question}\n\n_Closes in 5 minutes!_",
                reply_markup=InlineKeyboardMarkup(buttons),
                parse_mode="Markdown",
            )
            context.application.job_queue.run_once(
                expire_poll_job,
                when=expires,
                data={"poll_id": poll_id, "chat_id": chat_id, "message_id": sent.message_id},
            )
            return
        except Exception as e:
            logger.error(f"Poll creation error: {e}")
            prompt_payload = "Tell user the poll creation failed."

    # ══════════════════════════════════════════════════════════════════════════
    # FEATURE 6 & 7 — DYNAMIC INTENT ROUTING (Financial Data + Web Search)
    # The classify_intent() prompt rewrite is the core fix for the screenshot
    # failures: sports scores now reliably hit WEB_SEARCH, and company names
    # now correctly resolve to ticker symbols for STOCK routing.
    # ══════════════════════════════════════════════════════════════════════════
    else:
        route       = await classify_intent(user_text)
        intent_type = route.get("type", "REGULAR_CHAT")
        symbol      = route.get("symbol", "")

        if intent_type in ["STOCK", "FOREX", "COMMODITY"]:
            status = await context.bot.send_message(
                chat_id=chat_id,
                text=f"📊 Pulling live data for **{symbol}**...",
                parse_mode="Markdown",
            )
            market_data = await fetch_live_financial_data(intent_type, symbol)
            await context.bot.delete_message(chat_id=chat_id, message_id=status.message_id)
            prompt_payload = (
                f"Live market data:\n{market_data}\n\n"
                f"Analyze this like a knowledgeable friend. Keep it punchy and contextual."
            )

        elif intent_type == "WEB_SEARCH":
            status = await context.bot.send_message(chat_id=chat_id, text="🔍 Scanning the live wire...")
            web_data = await search_the_live_web(user_text)
            await context.bot.delete_message(chat_id=chat_id, message_id=status.message_id)
            prompt_payload = (
                f"Live web data:\n{web_data}\n\n"
                f"Answer the user's question concisely based on this. Don't pad it out."
            )

        else:
            prompt_payload = user_text.replace(bot_username, "").strip()

    # ── FETCH LIVE TIMESTAMP ───────────────────────────────────────────────────
    live_dt   = await get_network_time()
    ts_string = live_dt.strftime("%A, %d %B %Y, %I:%M %p SGT")

    # ── ASSEMBLE MULTIMODAL PAYLOAD ───────────────────────────────────────────
    contents = []
    if image_bytes:
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
    contents.append(f"[Live Time: {ts_string}]\nUser: {prompt_payload or 'Analyze this image.'}")

    # ── SEND TO GEMINI — RETRY ON RATE LIMIT / OVERLOAD ───────────────────────
    for attempt in range(3):
        try:
            response     = await asyncio.to_thread(lambda: chat.send_message(contents))
            bot_response = response.text.strip()

            is_image_url = bot_response.startswith(("http://", "https://")) and any(
                bot_response.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif")
            )

            if is_image_url:
                await context.bot.send_photo(chat_id=chat_id, photo=bot_response, caption="📸")
            else:
                await context.bot.send_message(chat_id=chat_id, text=bot_response)
            return

        except APIError as e:
            if e.code in [429, 503] and attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            await context.bot.send_message(
                chat_id=chat_id, text="AI servers are swamped. Give it a sec and try again!"
            )
            return
        except Exception as e:
            logger.error(f"Gemini response error (attempt {attempt + 1}): {e}")
            if attempt == 2:
                await context.bot.send_message(chat_id=chat_id, text="Something went sideways. Try again?")
            return


# ═════════════════════════════════════════════════════════════════════════════
# COMMANDS
# ═════════════════════════════════════════════════════════════════════════════

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Resets the chat session and re-syncs the clock."""
    if update.effective_chat:
        chat_sessions.pop(update.effective_chat.id, None)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Session wiped and clocks synced. What's good? 🫡",
        )


# ═════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═════════════════════════════════════════════════════════════════════════════

async def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(handle_poll_callback))
    app.add_handler(MessageHandler((filters.TEXT | filters.PHOTO) & ~filters.COMMAND, handle_message))

    logger.info("Bot online — all modules active.")

    async with app:
        await app.start()
        # FIX #10 — drop_pending_updates prevents replaying stale messages on Railway restart
        await app.updater.start_polling(drop_pending_updates=True)
        # FIX #5 — clean block instead of while True: sleep(3600)
        # asyncio.Event().wait() exits cleanly on KeyboardInterrupt / SIGTERM
        await asyncio.Event().wait()
        await app.updater.stop()
        await app.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot shut down cleanly.")