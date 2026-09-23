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
  | { type: "error"; message: string };

interface Pending {
  answer_id: string;
  question_id: number;
  choice: number;
}

export interface GameState {
  public: Snapshot;
  you: PrivateView | null;
}

// A restarted server starts a new epoch; within one epoch, higher seq wins.
function isNewer(next: Snapshot, prev: Snapshot | undefined): boolean {
  return !prev || next.epoch !== prev.epoch || next.seq > prev.seq;
}

export function useGameSocket(server: string, code: string, name: string) {
  const [state, setState] = useState<GameState | null>(null);
  const socket = useRef<WebSocket | null>(null);
  const pendingKey = `pending:${code}`;
  const pending = useRef<Pending[]>(
    JSON.parse(localStorage.getItem(pendingKey) ?? "[]"),
  );

  const savePending = () =>
    localStorage.setItem(pendingKey, JSON.stringify(pending.current));
  const send = (a: Pending) =>
    socket.current?.readyState === WebSocket.OPEN &&
    socket.current.send(JSON.stringify({ type: "answer", ...a }));

  useEffect(() => {
    let attempt = 0;
    let stopped = false;

    function connect() {
      const player = localStorage.getItem(`player:${code}`);
      const who = player ? `player=${player}` : `name=${encodeURIComponent(name)}`;
      const ws = new WebSocket(`${server}/ws/${code}?${who}`);
      socket.current = ws;

      ws.onopen = () => {
        attempt = 0;
        pending.current.forEach(send); // replay anything never acknowledged
      };
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data) as ServerMessage;
        if (msg.type === "welcome") {
          localStorage.setItem(`player:${code}`, msg.player);
        } else if (msg.type === "ack") {
          pending.current = pending.current.filter(
            (a) => a.answer_id !== msg.answer_id,
          );
          savePending();
        } else if (msg.type === "state") {
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
    const a = { answer_id: crypto.randomUUID(), question_id, choice };
    pending.current.push(a);
    savePending(); // survives a reload or a dead battery mid-question
    send(a);
  }

  return { state, answer };
}
