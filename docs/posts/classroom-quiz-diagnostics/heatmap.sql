-- One row per (session, objective): the heatmap cell and its denominators.
-- Reads Part 1's questions and Part 2's session tables; same numbers as
-- objective_cells() in diagnostics.py.

create view objective_cells as
with answered as (
  select r.session_id, r.player_id, r.timing, q.objective_id,
         (q.options -> r.choice ->> 'correct')::boolean as correct
  from responses r
  join questions q on q.id = r.question_id
),
per_student as (  -- each student counts once per cell, however many items
  select session_id, objective_id, player_id, avg(correct::int) as share
  from answered
  where timing <> 'late_after_reveal'  -- the key was already on the screen
  group by 1, 2, 3
),
scored as (
  select session_id, objective_id, count(*) as students, avg(share) as mastery
  from per_student
  group by 1, 2
),
expected as (  -- answers we could have had: who was there when each question
               -- opened, plus anyone who joined late and answered it anyway
  select seats.session_id, q.objective_id, count(*) as possible
  from (
    select sq.session_id, sq.question_id, sp.player_id
    from session_questions sq
    join session_players sp
      on sp.session_id = sq.session_id and sp.joined_at <= sq.opened_at
    union
    select session_id, question_id, player_id from responses
  ) seats
  join questions q on q.id = seats.question_id
  group by 1, 2
),
counts as (
  select session_id, objective_id,
         count(*) filter (where timing <> 'late_after_reveal') as usable,
         count(*) filter (where timing = 'late_after_reveal') as late_after_reveal
  from answered
  group by 1, 2
)
select e.session_id, e.objective_id,
       coalesce(s.students, 0) as students,
       s.mastery,
       w.low, w.high,
       coalesce(c.usable, 0)::float / e.possible as answer_rate,
       coalesce(c.late_after_reveal, 0) as late_after_reveal
from expected e
left join scored s using (session_id, objective_id)
left join counts c using (session_id, objective_id)
left join lateral (  -- Wilson interval, z = 1.96; nulls when nobody answered
  select greatest(0, (p + z*z/(2*n) - z*sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)) as low,
         least(1, (p + z*z/(2*n) + z*sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)) as high
  from (select s.mastery::float as p, nullif(s.students, 0)::float as n, 1.96 as z) v
) w on true;
