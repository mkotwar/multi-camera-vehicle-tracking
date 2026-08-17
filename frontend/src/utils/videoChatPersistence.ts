import type { ChatMessage, PersistedVideoChatSession } from "../types/videoChat";

const ACTIVE_RUN_KEY = "video-chat.active-run-id.v1";
const ACTIVE_RUNS_KEY = "video-chat.active-run-ids.v1";
const SESSIONS_BY_RUN_KEY = "video-chat.sessions-by-run.v1";

export function scopeKeyForRunIds(runIds: string[]): string {
  const cleaned = runIds.map((item) => item.trim()).filter(Boolean);
  return cleaned.length ? cleaned.join("|") : "latest";
}

export function createVideoChatSession(runId: string, runIds?: string[]): PersistedVideoChatSession {
  const now = new Date().toISOString();
  const scopedRunIds = runIds?.length ? runIds : [runId];
  return {
    version: 1,
    session_id: `video-chat-${scopeKeyForRunIds(scopedRunIds)}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    run_id: runId,
    run_ids: scopedRunIds,
    created_at: now,
    updated_at: now,
    messages: [],
  };
}

export function loadActiveVideoChatRunIds(): string[] | null {
  if (!canUseLocalStorage()) return null;
  try {
    const raw = window.localStorage.getItem(ACTIVE_RUNS_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return null;
    const runIds = parsed.map((item) => String(item).trim()).filter(Boolean);
    return runIds.length ? runIds : null;
  } catch {
    return null;
  }
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
  saveActiveVideoChatRunIds([runId]);
}

export function saveActiveVideoChatRunIds(runIds: string[]): void {
  if (!canUseLocalStorage()) return;
  try {
    const cleaned = runIds.map((item) => item.trim()).filter(Boolean);
    const primary = cleaned[0] ?? "latest";
    window.localStorage.setItem(ACTIVE_RUN_KEY, primary);
    window.localStorage.setItem(ACTIVE_RUNS_KEY, JSON.stringify(cleaned.length ? cleaned : [primary]));
  } catch {
    // Persistence must never make the chat unusable.
  }
}

export function loadVideoChatSessionForRun(runId: string): PersistedVideoChatSession | null {
  return loadVideoChatSessionForRunIds([runId]);
}

export function loadVideoChatSessionForRunIds(runIds: string[]): PersistedVideoChatSession | null {
  if (!canUseLocalStorage()) return null;
  const sessions = loadSessionMap();
  const session = sessions[scopeKeyForRunIds(runIds)];
  return isPersistedVideoChatSession(session) ? session : null;
}

export function saveVideoChatSessionForRun(session: PersistedVideoChatSession): void {
  if (!canUseLocalStorage()) return;
  try {
    const nextSession = { ...session, updated_at: new Date().toISOString() };
    const sessions = loadSessionMap();
    const key = scopeKeyForRunIds(nextSession.run_ids?.length ? nextSession.run_ids : [nextSession.run_id]);
    window.localStorage.setItem(SESSIONS_BY_RUN_KEY, JSON.stringify({ ...sessions, [key]: nextSession }));
    saveActiveVideoChatRunIds(nextSession.run_ids?.length ? nextSession.run_ids : [nextSession.run_id]);
  } catch {
    // Persistence must never make the chat unusable.
  }
}

export function replaceVideoChatSessionForRun(runId: string): PersistedVideoChatSession {
  const nextSession = createVideoChatSession(runId);
  saveVideoChatSessionForRun(nextSession);
  return nextSession;
}

export function replaceVideoChatSessionForRunIds(runIds: string[]): PersistedVideoChatSession {
  const cleaned = runIds.length ? runIds : ["latest"];
  const nextSession = createVideoChatSession(cleaned[0], cleaned);
  saveVideoChatSessionForRun(nextSession);
  return nextSession;
}

export function getOrCreateVideoChatSessionForRun(runId: string): PersistedVideoChatSession {
  return loadVideoChatSessionForRun(runId) ?? createVideoChatSession(runId);
}

export function getOrCreateVideoChatSessionForRunIds(runIds: string[]): PersistedVideoChatSession {
  const cleaned = runIds.length ? runIds : ["latest"];
  return loadVideoChatSessionForRunIds(cleaned) ?? createVideoChatSession(cleaned[0], cleaned);
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
  if (session.run_ids !== undefined && (!Array.isArray(session.run_ids) || !session.run_ids.every((item) => typeof item === "string" && item.trim()))) return false;
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
    for (const [key, session] of Object.entries(parsed as Record<string, unknown>)) {
      if (isPersistedVideoChatSession(session)) {
        sessions[key] = session;
      }
    }
    return sessions;
  } catch {
    return {};
  }
}
