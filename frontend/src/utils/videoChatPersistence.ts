import type { ChatMessage, PersistedVideoChatSession } from "../types/videoChat";

const ACTIVE_RUN_KEY = "video-chat.active-run-id.v1";
const SESSIONS_BY_RUN_KEY = "video-chat.sessions-by-run.v1";

export function createVideoChatSession(runId: string): PersistedVideoChatSession {
  const now = new Date().toISOString();
  return {
    version: 1,
    session_id: `video-chat-${runId}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    run_id: runId,
    created_at: now,
    updated_at: now,
    messages: [],
  };
}

export function loadActiveVideoChatRunId(): string | null {
  if (!canUseLocalStorage()) return null;
  try {
    const runId = window.localStorage.getItem(ACTIVE_RUN_KEY);
    return runId?.trim() || null;
  } catch {
    return null;
  }
}

export function saveActiveVideoChatRunId(runId: string): void {
  if (!canUseLocalStorage()) return;
  try {
    window.localStorage.setItem(ACTIVE_RUN_KEY, runId);
  } catch {
    // Persistence must never make the chat unusable.
  }
}

export function loadVideoChatSessionForRun(runId: string): PersistedVideoChatSession | null {
  if (!canUseLocalStorage()) return null;
  const sessions = loadSessionMap();
  const session = sessions[runId];
  return isPersistedVideoChatSession(session) ? session : null;
}

export function saveVideoChatSessionForRun(session: PersistedVideoChatSession): void {
  if (!canUseLocalStorage()) return;
  try {
    const nextSession = { ...session, updated_at: new Date().toISOString() };
    const sessions = loadSessionMap();
    window.localStorage.setItem(SESSIONS_BY_RUN_KEY, JSON.stringify({ ...sessions, [nextSession.run_id]: nextSession }));
    window.localStorage.setItem(ACTIVE_RUN_KEY, nextSession.run_id);
  } catch {
    // Persistence must never make the chat unusable.
  }
}

export function replaceVideoChatSessionForRun(runId: string): PersistedVideoChatSession {
  const nextSession = createVideoChatSession(runId);
  saveVideoChatSessionForRun(nextSession);
  return nextSession;
}

export function getOrCreateVideoChatSessionForRun(runId: string): PersistedVideoChatSession {
  return loadVideoChatSessionForRun(runId) ?? createVideoChatSession(runId);
}

function canUseLocalStorage() {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

function isPersistedVideoChatSession(value: unknown): value is PersistedVideoChatSession {
  if (!value || typeof value !== "object") return false;
  const session = value as Partial<PersistedVideoChatSession>;
  if (session.version !== 1) return false;
  if (typeof session.session_id !== "string" || !session.session_id.trim()) return false;
  if (typeof session.run_id !== "string" || !session.run_id.trim()) return false;
  if (typeof session.created_at !== "string" || typeof session.updated_at !== "string") return false;
  if (!Array.isArray(session.messages)) return false;
  return session.messages.every(isChatMessage);
}

function isChatMessage(value: unknown): value is ChatMessage {
  if (!value || typeof value !== "object") return false;
  const item = value as Record<string, unknown>;
  return (
    typeof item.id === "string" &&
    (item.role === "user" || item.role === "assistant") &&
    typeof item.text === "string" &&
    typeof item.timestamp === "string"
  );
}

function loadSessionMap(): Record<string, PersistedVideoChatSession> {
  try {
    const raw = window.localStorage.getItem(SESSIONS_BY_RUN_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    const sessions: Record<string, PersistedVideoChatSession> = {};
    for (const [runId, session] of Object.entries(parsed as Record<string, unknown>)) {
      if (isPersistedVideoChatSession(session) && session.run_id === runId) {
        sessions[runId] = session;
      }
    }
    return sessions;
  } catch {
    return {};
  }
}
