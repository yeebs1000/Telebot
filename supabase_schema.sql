-- Run this in the Supabase SQL editor (Project → SQL Editor → New query)
-- before starting the bot. Requires the pgvector extension for embeddings.

create extension if not exists vector;

-- Raw group chat history, used for /summarize and personality lookups.
create table if not exists group_chat_logs (
    id          bigint generated always as identity primary key,
    chat_id     bigint not null,
    sender      text not null,
    message     text not null,
    created_at  timestamptz not null default now()
);

-- Vector embeddings of group messages, used for semantic memory search.
-- Gemini's text-embedding-004 natively produces 768-dim vectors; the OpenAI
-- provider requests 768 dims via the `dimensions` param to match. This table
-- stays empty (and the feature is skipped) while AI_PROVIDER=claude, since
-- Anthropic has no embeddings API.
create table if not exists group_embeddings (
    id          bigint generated always as identity primary key,
    chat_id     bigint not null,
    sender      text not null,
    message     text not null,
    embedding   vector(768) not null,
    created_at  timestamptz not null default now()
);

-- Live inline-keyboard polls with a 5-minute expiry window.
create table if not exists active_polls (
    poll_id     text primary key,
    chat_id     bigint not null,
    question    text not null,
    options     jsonb not null,
    votes       jsonb not null default '{}'::jsonb,
    expires_at  timestamptz not null
);

-- RPC used by the "search memory" feature for cosine-similarity lookup.
create or replace function match_chat_embeddings(
    query_embedding vector(768),
    match_threshold float,
    match_count int,
    filter_chat_id bigint
)
returns table (
    sender text,
    message text,
    similarity float
)
language sql stable
as $$
    select
        sender,
        message,
        1 - (embedding <=> query_embedding) as similarity
    from group_embeddings
    where chat_id = filter_chat_id
      and 1 - (embedding <=> query_embedding) > match_threshold
    order by embedding <=> query_embedding
    limit match_count;
$$;
