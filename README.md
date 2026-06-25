# AI-Powered Telebot

An intelligent Telegram group assistant that works with the AI provider of
your choice — Gemini, OpenAI, or Claude — picked with a single environment
variable. It chats naturally, remembers what your group talks about, and can
pull live stock, forex, commodity, and web-search data into the conversation.

## Features

- **Conversational chat** — mention the bot (or DM it) and it replies in a
  casual, friend-like tone.
- **Bring your own AI provider** — switch between Gemini, OpenAI, and Claude
  with one env var; no code changes needed. See
  [Choosing your AI provider](#choosing-your-ai-provider) below.
- **Group chat logging & summaries** — ask it to "summarize" and it recaps
  recent group activity from a Supabase-backed log.
- **Personality lookups** — "what do you think of \<name>" pulls that
  person's message history and gives a light-hearted read on them.
- **Semantic memory search** — "where did we..." / "search memory ..."
  searches past messages by meaning (vector embeddings), not just keywords.
  Requires a provider that supports embeddings (Gemini or OpenAI).
- **Natural-language reminders** — "remind me to ... at 6pm" schedules a
  one-off reminder via the bot's job queue.
- **Live polls** — `poll: Question | Option 1 | Option 2` creates an inline
  voting poll that auto-closes and tallies after 5 minutes.
- **Live market data** — ask about a stock, forex pair, or commodity and it
  fetches a real-time quote (Alpha Vantage).
- **Live web search** — sports scores, breaking news, and anything else that
  needs current information is routed to a web search (Tavily) instead of
  the model's static knowledge.
- **Image understanding** — send a photo in a chat where the bot is
  mentioned and it will analyze it.

An intent classifier (a small AI call) decides, per message, whether to
route to chat, market data, or web search — no slash commands needed for
those features.

## Choosing your AI provider

Set `AI_PROVIDER` in `.env` to one of `gemini`, `openai`, or `claude`, and
fill in the matching API key. Everything else in the bot — chat, intent
routing, reminders, market data, polls — works identically regardless of
provider.

| Provider | `AI_PROVIDER` value | API key env var     | Chat | Intent routing | Embeddings / semantic memory |
|----------|----------------------|----------------------|------|-----------------|-------------------------------|
| Gemini   | `gemini`             | `GEMINI_API_KEY`     | ✅   | ✅ (native JSON mode) | ✅ (native 768-dim)            |
| OpenAI   | `openai`              | `OPENAI_API_KEY`     | ✅   | ✅ (native JSON mode) | ✅ (768-dim via `dimensions`)  |
| Claude   | `claude`              | `ANTHROPIC_API_KEY`  | ✅   | ✅ (prompted JSON)    | ❌ — no embeddings API; semantic memory search is automatically disabled |

You can also override the default model per provider with `AI_MODEL`
(defaults: `gemini-3.1-flash-lite`, `gpt-4o-mini`, `claude-sonnet-4-6`).

Only the SDK for your chosen provider is actually used at runtime — the
other two packages in `requirements.txt` just sit unused, so switching
providers later is a one-line `.env` change, no reinstall required.

Want to add a provider that isn't listed (Mistral, Llama, a local model,
etc.)? See [Adding a new AI provider](CONTRIBUTING.md#adding-a-new-ai-provider)
in CONTRIBUTING.md — it's a single new file implementing one small interface.

## Requirements

- Python 3.11+
- A [Telegram bot token](https://core.telegram.org/bots#how-do-i-create-a-bot)
  from [@BotFather](https://t.me/BotFather)
- An API key for whichever AI provider you pick:
  [Gemini](https://aistudio.google.com/apikey),
  [OpenAI](https://platform.openai.com/api-keys), or
  [Anthropic](https://console.anthropic.com/settings/keys)
- A free [Supabase](https://supabase.com) project (used for chat logs,
  semantic memory, and live polls)
- Optional: a [Tavily](https://tavily.com) API key for live web search
- Optional: an [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
  API key for stock/forex/commodity quotes

## Setup

1. **Clone and install dependencies**

   ```bash
   git clone https://github.com/yeebs1000/ai-powered-telebot.git
   cd ai-powered-telebot
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Set up Supabase**

   Create a project at [supabase.com](https://supabase.com), open the
   **SQL Editor**, and run the contents of [`supabase_schema.sql`](supabase_schema.sql).
   This creates the tables and RPC function the bot needs (chat logs,
   embeddings, and live polls).

3. **Configure environment variables**

   ```bash
   cp .env.example .env
   ```

   Fill in `.env` with your keys:

   | Variable             | Required | Purpose                                   |
   |----------------------|----------|---------------------------------------------|
   | `AI_PROVIDER`        | Yes      | `gemini`, `openai`, or `claude`             |
   | `AI_MODEL`           | No       | Override the default model for your provider |
   | `TELEGRAM_BOT_TOKEN` | Yes      | Bot auth token from @BotFather              |
   | `GEMINI_API_KEY`     | If using Gemini | Chat, routing, embeddings             |
   | `OPENAI_API_KEY`     | If using OpenAI | Chat, routing, embeddings             |
   | `ANTHROPIC_API_KEY`  | If using Claude | Chat, routing                         |
   | `SUPABASE_URL`       | Yes      | Supabase project URL                        |
   | `SUPABASE_KEY`       | Yes      | Supabase service/anon key                   |
   | `TAVILY_API_KEY`     | No       | Enables live web search                     |
   | `ALPHA_VANTAGE_KEY`  | No       | Enables stock/forex/commodity quotes         |

4. **Run it**

   ```bash
   python main.py
   ```

   Add the bot to a Telegram group (or message it directly), mention it by
   `@username`, and start chatting.

## Deploying

The repo includes a `Procfile` and `runtime.txt` for platforms like
[Railway](https://railway.app) or Heroku-style buildpacks:

1. Push the repo to your deployment platform.
2. Set the same environment variables from `.env.example` in the platform's
   config/secrets UI.
3. Deploy — it runs as a long-lived worker process (`python main.py`),
   polling Telegram for updates.

## How it works

`main.py` is a single-file bot built on `python-telegram-bot`. AI calls go
through a small adapter layer in `providers/` (see
[providers/base.py](providers/base.py)) so the rest of the bot never talks
to a specific vendor SDK directly. Every incoming message goes through
`handle_message`, which:

1. Logs non-directed group messages (and their embeddings, if the active
   provider supports them) to Supabase in the background, for later
   summaries/search.
2. If the bot is mentioned (or it's a DM), checks for built-in features
   (summarize, personality lookup, reminders, memory search, polls) by
   simple keyword matching.
3. Otherwise calls `classify_intent()` — a lightweight AI call that returns
   structured JSON to route the message to a stock/forex/commodity lookup,
   a live web search, or plain conversation.
4. Sends the assembled context (plus a live timestamp, and an image if one
   was attached) to a per-chat AI session and replies.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
