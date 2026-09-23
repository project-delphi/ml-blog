import { useEffect, useRef, useState } from "react";

export type Phase =
  | "lobby"
  | "question"
  | "locked"
  | "reveal"
  | "leaderboard"
  | "final";

export interface Snapshot {
  room: string;
  epoch: string;
  seq: number;
  phase: Phase;
  players: number;
  question?: {
    id: number;
    stem: string;
    options: string[];
    remaining_ms?: number; // counted down locally; the server locks on its own clock
    answer?: number; // present only from the reveal on
    counts?: number[];
    explanation?: string;
    evidence?: string;
    source?: string;
  };
  leaderboard?: { name: string; score: number }[];
}

export interface PrivateView {
  player: string;
  score: number;
  answered?: boolean;
  correct?: boolean;
  points?: number;
}

type ServerMessage =
  | { type: "welcome"; player: string }
  | { type: "state"; public: Snapshot; you: PrivateView | null }
  | { type: "ack"; answer_id: string }
  | { type: "error"; message: string; answer_id?: string };

interface Pending {
  answer_id: string;
  question_id: number;
  choice: number;
  epoch: string; // the game it belongs to; room codes get reused
}

export interface GameState {
  public: Snapshot;
  you: PrivateView | null;
}

// A restarted server starts a new epoch; within one epoch, higher seq wins.
function isNewer(next: Snapshot, prev: Snapshot | undefined): boolean {
  return !prev || next.epoch !== prev.epoch || next.seq > prev.seq;
}

// The phone names itself once per room code, so every reconnect, even one
// before the server's `welcome` arrived, comes back as the same player.
function playerId(code: string): string {
  const key = `player:${code}`;
  const id = localStorage.getItem(key) ?? crypto.randomUUID();
  localStorage.setItem(key, id);
  return id;
}

export function useGameSocket(server: string, code: string, name: string) {
  const [state, setState] = useState<GameState | null>(null);
  const socket = useRef<WebSocket | null>(null);
  const epoch = useRef<string | null>(null);
  const pendingKey = `pending:${code}`;
  const pending = useRef<Pending[]>(
    JSON.parse(localStorage.getItem(pendingKey) ?? "[]"),
  );

  const savePending = () =>
    localStorage.setItem(pendingKey, JSON.stringify(pending.current));
  const drop = (answer_id: string) => {
    pending.current = pending.current.filter((a) => a.answer_id !== answer_id);
    savePending();
  };
  const send = (a: Pending) =>
    socket.current?.readyState === WebSocket.OPEN &&
    socket.current.send(JSON.stringify({ type: "answer", ...a }));

  useEffect(() => {
    let attempt = 0;
    let stopped = false;

    function connect() {
      const who = `player=${playerId(code)}&name=${encodeURIComponent(name)}`;
      const ws = new WebSocket(`${server}/ws/${code}?${who}`);
      let replayed = false;
      socket.current = ws;

      ws.onopen = () => {
        attempt = 0;
      };
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data) as ServerMessage;
        if (msg.type === "ack") {
          drop(msg.answer_id);
        } else if (msg.type === "error" && msg.answer_id) {
          drop(msg.answer_id); // refused for good: retrying won't help
        } else if (msg.type === "state") {
          epoch.current = msg.public.epoch;
          if (!replayed) {
            // Now that we know which game this is, forget answers from any
            // other one, and replay the rest.
            replayed = true;
            pending.current = pending.current.filter(
              (a) => a.epoch === msg.public.epoch,
            );
            savePending();
            pending.current.forEach(send);
          }
          setState((prev) =>
            isNewer(msg.public, prev?.public) ? { public: msg.public, you: msg.you } : prev,
          );
        }
      };
      ws.onclose = () => {
        if (!stopped) setTimeout(connect, Math.min(500 * 2 ** attempt++, 8000));
      };
    }

    connect();
    return () => {
      stopped = true;
      socket.current?.close();
    };
  }, [server, code, name]);

  function answer(question_id: number, choice: number) {
    if (!epoch.current) return; // no question can be showing yet
    const a = { answer_id: crypto.randomUUID(), question_id, choice, epoch: epoch.current };
    pending.current.push(a);
    savePending(); // survives a reload or a dead battery mid-question
    send(a);
  }

  return { state, answer };
}
