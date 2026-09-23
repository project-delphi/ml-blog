// @ts-check
import { useEffect, useRef } from "react";

/** @typedef {import("./useGameSocket").Phase} Phase */
/** @typedef {import("./useGameSocket").Snapshot} Snapshot */

/** Every phase change fires one cue. Only the projector plays sound. */
/** @type {Record<Phase, {theme: string, sound: string, loop?: boolean}>} */
const CUES = {
  lobby: { theme: "calm", sound: "/audio/lobby.mp3", loop: true },
  question: { theme: "tense", sound: "/audio/countdown.mp3" },
  locked: { theme: "tense", sound: "/audio/buzzer.mp3" },
  reveal: { theme: "bright", sound: "/audio/reveal.mp3" },
  leaderboard: { theme: "party", sound: "/audio/drumroll.mp3" },
  final: { theme: "party", sound: "/audio/fanfare.mp3", loop: true },
};

/** @param {{ snapshot: Snapshot, qrUrl: string }} props */
export function Projector({ snapshot, qrUrl }) {
  const { phase, question } = snapshot;
  const cue = CUES[phase];
  const audio = useRef(/** @type {HTMLAudioElement | null} */ (null));

  useEffect(() => {
    audio.current?.pause();
    audio.current = new Audio(cue.sound);
    audio.current.loop = cue.loop ?? false;
    audio.current.play().catch(() => {}); // blocked until someone clicks once
  }, [phase, question?.id]);

  return (
    // A new key per phase remounts the stage, which replays its CSS entry
    // animation; data-theme swaps the colour tokens the stylesheet defines.
    <main key={`${phase}-${question?.id}`} className="stage" data-theme={cue.theme}>
      {phase === "lobby" && <img className="qr" src={qrUrl} alt="Scan to join" />}
      {question && <h1>{question.stem}</h1>}
      {question?.options.map((text, i) => (
        <div key={i} className={i === question.answer ? "option right" : "option"}>
          {text}
          {question.counts && <span className="count">{question.counts[i]}</span>}
        </div>
      ))}
      {phase === "reveal" && question?.evidence && (
        <blockquote>
          {question.evidence} <cite>{question.source}</cite>
        </blockquote>
      )}
      {snapshot.leaderboard?.map((row, rank) => (
        <div key={rank} className="row">
          {rank + 1}. {row.name} <b>{row.score}</b>
        </div>
      ))}
    </main>
  );
}
