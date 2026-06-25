# Contributing to AI-Powered Telebot

Thanks for considering a contribution. A few things to know before you start.

## Setup

Follow the steps in [README.md](README.md): create a Supabase project, run
`supabase_schema.sql`, copy `.env.example` to `.env`, and fill in at least
the required keys (`AI_PROVIDER` + its matching API key, `TELEGRAM_BOT_TOKEN`,
`SUPABASE_URL`, `SUPABASE_KEY`).

## Running it locally

```bash
python main.py
```

Message your bot in a private chat, or add it to a test Telegram group, to
exercise changes end-to-end.

## Testing changes — please read this before opening a PR

This bot calls paid, metered APIs (your chosen AI provider, and optionally
Tavily / Alpha Vantage) for almost everything it does. **Don't spam test
messages against a live bot token just to "check" a change** — each mention
triggers at least one AI call, often more (routing classification,
embeddings, etc.).

- For logic-only changes (parsing, formatting, routing keyword matching),
  test the relevant function directly in a Python shell rather than running
  the full bot.
- If you need to verify an end-to-end flow, use a fresh API key with a free
  tier (Gemini has one) rather than a production key.
- If your change touches `providers/`, test it against whichever provider(s)
  it affects — a change to the Gemini adapter doesn't need to be re-tested
  against OpenAI/Claude, and vice versa.
- There is no automated test suite — CI only does a Python syntax/import
  compile check (`python -m compileall`). Manual verification against a
  test bot/group is expected before opening a PR.

## Code style

- Single-file bot (`main.py`) — keep new features as additional `elif`
  branches in `handle_message` or as new async helper functions, following
  the existing pattern (feature comment headers, `asyncio.to_thread()` for
  any blocking Supabase calls).
- AI calls go through `providers/`, never directly through a vendor SDK in
  `main.py` — that's what keeps the bot provider-agnostic.
- No hardcoded secrets, ever — everything configurable goes through
  `os.getenv()` and gets documented in `.env.example`.

## Adding a new AI provider

The bot only depends on the small interface in
[`providers/base.py`](providers/base.py): an `AIProvider` with
`create_chat()` / `generate_text()` / `generate_json()` / (optionally)
`embed()`, and a `ChatSession` with a single `send(text, image=None)` method.
Look at [`providers/claude_provider.py`](providers/claude_provider.py) for
the simplest example (no native JSON mode, no embeddings) or
[`providers/gemini_provider.py`](providers/gemini_provider.py) for one with
both.

To add a provider (e.g. Mistral, a local model server, etc.):

1. Create `providers/<name>_provider.py` with a `<Name>ChatSession(ChatSession)`
   and a `<Name>Provider(AIProvider)` implementing the abstract methods.
   Set `supports_embeddings = True` only if you also implement `embed()` and
   it returns vectors compatible with the `vector(768)` column in
   `supabase_schema.sql` (or note in your PR that the schema needs a
   matching dimension change).
2. Register it in [`providers/__init__.py`](providers/__init__.py)'s
   `get_provider()` — add an `elif name == "<name>":` branch that reads the
   relevant API key from the environment and constructs your class.
3. Document the new `AI_PROVIDER` value and its required env var(s) in
   `.env.example` and the provider table in `README.md`.
4. Test it manually against a test bot (see "Testing changes" above).

## Submitting changes

1. Fork the repo and create a feature branch.
2. Make your changes and verify manually against a test bot.
3. Open a PR describing what changed and why, and how you tested it.
