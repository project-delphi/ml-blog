-- Live sessions, extending Part 1's schema.sql. The game server writes these
-- rows; Part 3's diagnostics read them.

create table sessions (
  id            text primary key,      -- '<room code>-<epoch>'
  course_id     text not null references courses,
  question_ids  bigint[] not null,     -- in the order they were planned
  started_at    timestamptz not null default now()
);

-- Who was in the room and from when: the denominator for every answer rate.
create table session_players (
  session_id  text not null references sessions,
  player_id   text not null,
  joined_at   timestamptz not null,
  primary key (session_id, player_id)
);

create table session_questions (
  session_id   text   not null references sessions,
  question_id  bigint not null references questions,
  opened_at    timestamptz not null,
  primary key (session_id, question_id)
);

create table responses (
  answer_id    text primary key,       -- made on the phone; a replay is a no-op
  session_id   text   not null references sessions,
  player_id    text   not null,
  question_id  bigint not null references questions,
  choice       int    not null,
  timing       text   not null
               check (timing in ('on_time', 'late_before_reveal', 'late_after_reveal')),
  points       int    not null,
  received_at  timestamptz not null,
  unique (session_id, player_id, question_id),
  foreign key (session_id, player_id) references session_players
);
