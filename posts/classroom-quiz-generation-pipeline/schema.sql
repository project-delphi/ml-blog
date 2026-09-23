-- Every question keeps the address of the passage it came from and the record
-- of every review it went through. Plain Postgres; runs unchanged on Supabase.

create table courses (
  id        text primary key,
  title     text not null,
  language  text not null check (language in ('es', 'pt-BR', 'en'))
);

-- Objective ids are language-neutral, so a Spanish and an English section of
-- the same course aggregate onto the same rows.
create table objectives (
  id              text primary key,
  course_id       text not null references courses,
  statement       text not null,
  misconceptions  text[] not null  -- the closed list distractors are tagged from
);

create table documents (
  id           text primary key,
  course_id    text not null references courses,
  title        text not null,
  uploaded_at  timestamptz not null default now()
);

create table chunks (
  id           text primary key,  -- '<document>:p<page>:<sha256 prefix>'
  document_id  text not null references documents on delete cascade,
  page         int  not null,
  text         text not null
);

create type question_status as enum ('draft', 'verified', 'rejected', 'approved');

create table questions (
  id            bigint generated always as identity primary key,
  chunk_id      text not null references chunks,
  objective_id  text not null references objectives,
  bloom_level   text not null,
  status        question_status not null,
  stem          text,
  options       jsonb,  -- [{text, correct, misconception}]
  evidence      text,   -- verbatim from chunks.text
  explanation   text,
  created_at    timestamptz not null default now(),
  check (status = 'rejected' or evidence is not null)
);

-- One row per attempt: what the generator wrote and what the checks said.
create table reviews (
  question_id  bigint not null references questions on delete cascade,
  attempt      int    not null,
  problems     text[] not null,
  draft        jsonb,
  primary key (question_id, attempt)
);

-- Store one pipeline outcome (see outcome_payload in pipeline.py) in one call,
-- so a question never exists without its review history.
create function save_outcome(payload jsonb) returns bigint
language plpgsql as $$
declare
  qid bigint;
begin
  insert into questions
    (chunk_id, objective_id, bloom_level, status,
     stem, options, evidence, explanation)
  values
    (payload ->> 'chunk_id', payload ->> 'objective_id',
     payload ->> 'bloom_level', (payload ->> 'status')::question_status,
     payload #>> '{draft,stem}', payload #> '{draft,options}',
     payload #>> '{draft,evidence}', payload #>> '{draft,explanation}')
  returning id into qid;

  insert into reviews (question_id, attempt, problems, draft)
  select qid, a.n,
         array(select jsonb_array_elements_text(a.value -> 'problems')),
         nullif(a.value -> 'draft', 'null'::jsonb)
  from jsonb_array_elements(payload -> 'attempts') with ordinality as a(value, n);

  return qid;
end $$;
