create extension if not exists vector;

create schema if not exists vidhi;

create table if not exists vidhi.chunks (
  id           bigserial primary key,
  chunk_uid    text unique not null,
  content      text not null,
  heading_path text,
  chapter      text,
  page_start   int not null,
  page_end     int not null,
  token_count  int,
  embedding    public.vector(768),
  tsv tsvector generated always as (to_tsvector('english', content)) stored
);

create index if not exists chunks_embedding_idx
  on vidhi.chunks using hnsw (embedding vector_cosine_ops)
  with (m = 16, ef_construction = 64);

create index if not exists chunks_tsv_idx on vidhi.chunks using gin (tsv);

create table if not exists vidhi.traces (
  request_id   uuid primary key,
  query        text not null,
  route        text,
  sub_queries  jsonb,
  stages       jsonb,
  latency_ms   jsonb,
  tokens       jsonb,
  answer       text,
  citations    jsonb,
  grounded     boolean,
  abstained    boolean,
  degraded     boolean default false,
  error        text,
  created_at   timestamptz default now()
);

create index if not exists traces_created_idx on vidhi.traces (created_at desc);
