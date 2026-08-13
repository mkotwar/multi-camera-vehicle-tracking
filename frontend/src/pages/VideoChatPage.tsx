import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { ApiError } from "../api/client";
import { fetchRuns } from "../api/runs";
import { sendVideoChatMessage } from "../api/videoChat";
import { TrackDetailPanel, type TrackSelection } from "../components/track/TrackDetailPanel";
import type { RunSummary } from "../types/run";
import type { ChatMessage, EvidencePage, VideoChatResponse, VehicleEvidence } from "../types/videoChat";
import {
  getOrCreateVideoChatSessionForRun,
  loadActiveVideoChatRunId,
  replaceVideoChatSessionForRun,
  saveActiveVideoChatRunId,
  saveVideoChatSessionForRun,
} from "../utils/videoChatPersistence";
import { formatVideoTime } from "../utils/time";

const PROMPT_SUGGESTIONS = [
  "How many cars were there?",
  "Show white cars",
  "When were bikes more than cars?",
  "What colours were the motorcycles?",
];

export function VideoChatPage() {
  const [chatSession, setChatSession] = useState(() => getOrCreateVideoChatSessionForRun(loadActiveVideoChatRunId() ?? "latest"));
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [runId, setRunId] = useState(chatSession.run_id);
  const [messages, setMessages] = useState<ChatMessage[]>(chatSession.messages);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [userIsReading, setUserIsReading] = useState(false);
  const [selectedTrack, setSelectedTrack] = useState<TrackSelection | null>(null);
  const chatHistoryRef = useRef<HTMLDivElement | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    void fetchRuns()
      .then((items) => {
        setRuns(items);
        if (items[0]?.run_id && messages.length === 0 && chatSession.run_id === "latest") {
          loadRunSession(items[0].run_id);
        }
      })
      .catch(() => setRuns([]));
  }, []);

  useEffect(() => {
    if (!userIsReading) {
      endRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
    }
  }, [messages, isSending, userIsReading]);

  const selectedRun = useMemo(() => runs.find((item) => item.run_id === runId), [runId, runs]);
  const latestAssistantMessage = useMemo(() => [...messages].reverse().find((message) => message.role === "assistant"), [messages]);

  const handleScroll = () => {
    const element = chatHistoryRef.current;
    if (!element) return;
    const distanceFromBottom = element.scrollHeight - element.scrollTop - element.clientHeight;
    setUserIsReading(distanceFromBottom > 180);
  };

  const submitMessage = async (event?: FormEvent, override?: string) => {
    event?.preventDefault();
    const text = (override ?? input).trim();
    if (!text || isSending) return;
    const sessionId = chatSession.session_id;
    const userMessage: ChatMessage = { id: makeMessageId("user"), role: "user", text, timestamp: new Date().toISOString() };
    persistMessages((current) => [...current, userMessage]);
    setInput("");
    setError(null);
    setIsSending(true);
    try {
      const response = await sendVideoChatMessage({ message: text, run_id: runId, session_id: sessionId });
      persistMessages((current) => [...current, responseToMessage(response)]);
    } catch (chatError) {
      setError(describeChatError(chatError));
    } finally {
      setIsSending(false);
    }
  };

  const askForEvidence = () => {
    void submitMessage(undefined, "Show them");
  };

  const showMoreEvidence = () => {
    void submitMessage(undefined, "Show more");
  };

  const selectTrack = (track: TrackSelection) => {
    setSelectedTrack(track);
  };

  const startNewChat = (nextRunId = runId) => {
    const nextSession = replaceVideoChatSessionForRun(nextRunId);
    setChatSession(nextSession);
    setRunId(nextRunId);
    setMessages([]);
    setSelectedTrack(null);
    setInput("");
    setError(null);
    setUserIsReading(false);
  };

  const changeRun = (nextRunId: string) => {
    loadRunSession(nextRunId);
  };

  function loadRunSession(nextRunId: string) {
    const nextSession = getOrCreateVideoChatSessionForRun(nextRunId);
    setChatSession(nextSession);
    setRunId(nextRunId);
    setMessages(nextSession.messages);
    setSelectedTrack(null);
    setError(null);
    setUserIsReading(false);
    saveActiveVideoChatRunId(nextRunId);
    if (nextSession.messages.length === 0) {
      saveVideoChatSessionForRun(nextSession);
    }
  }

  function persistMessages(updater: (current: ChatMessage[]) => ChatMessage[]) {
    setMessages((current) => {
      const nextMessages = updater(current);
      const nextSession = { ...chatSession, run_id: runId, messages: nextMessages };
      setChatSession(nextSession);
      saveVideoChatSessionForRun(nextSession);
      return nextMessages;
    });
  }

  const runBadges = [
    { label: "Run", value: selectedRun?.run_id ?? runId },
    { label: "Cameras", value: selectedRun?.camera_count ?? "Any" },
    { label: "Duration", value: selectedRun?.duration_seconds != null ? formatVideoTime(selectedRun.duration_seconds) : "Saved run" },
    { label: "Vehicles", value: selectedRun?.track_count ?? "Unavailable" },
  ];

  return (
    <section className="page-stack video-chat-page">
      <section className="video-chat-hero">
        <div className="video-chat-title">
          <div className="video-chat-kicker">
            <span className="eyebrow">Video AI</span>
            <span className="ai-ready-indicator"><i aria-hidden="true" /> AI Ready</span>
          </div>
          <h1>Video Analytics Assistant</h1>
          <p>Ask questions about detected vehicles, colours, timing, and evidence.</p>
        </div>
        <div className="run-badge-grid" aria-label="Current run context">
          {runBadges.map((badge) => (
            <div key={badge.label} className={`run-badge ${badge.label === "Run" ? "active" : ""}`}>
              <span>{badge.label}</span>
              <strong>{badge.value}</strong>
            </div>
          ))}
        </div>
      </section>

      <section className="video-chat-workspace">
        <section className="chat-panel" aria-label="Video analytics assistant">
          <div className="chat-panel-toolbar">
            <label>
              <span>Run / video</span>
              <select value={runId} onChange={(event) => changeRun(event.target.value)} aria-label="Video chat run">
                {runs.length === 0 ? <option value="latest">Latest</option> : null}
                {runs.map((run) => (
                  <option key={run.run_id} value={run.run_id}>
                    {run.run_id}
                  </option>
                ))}
              </select>
            </label>
            <span className="status status-completed">{selectedRun?.status ?? "Latest"}</span>
            <button type="button" className="secondary-button compact-action" onClick={() => startNewChat()}>
              New Chat
            </button>
          </div>

          <div ref={chatHistoryRef} className="chat-history" aria-label="Video analytics chat history" onScroll={handleScroll}>
            {messages.length === 0 ? <ChatEmptyState onPrompt={submitMessage} /> : null}
            {messages.map((message) => (
              <ChatMessageCard
                key={message.id}
                message={message}
                onShowEvidence={askForEvidence}
                onShowMore={showMoreEvidence}
                onSelectTrack={selectTrack}
                selectedTrack={selectedTrack}
              />
            ))}
            {isSending ? <TypingState /> : null}
            <div ref={endRef} />
          </div>

          {error ? <div className="chat-error" role="alert">{error}</div> : null}

          <ChatComposer
            input={input}
            isSending={isSending}
            onInput={setInput}
            onSubmit={submitMessage}
          />
        </section>

        <VideoContextPanel
          selectedRun={selectedRun}
          runId={runId}
          latestMessage={latestAssistantMessage}
          selectedTrack={selectedTrack}
        />
      </section>
    </section>
  );
}

function responseToMessage(response: VideoChatResponse): ChatMessage {
  return {
    id: makeMessageId("assistant"),
    role: "assistant",
    text: response.answer,
    timestamp: new Date().toISOString(),
    evidence: response.evidence,
    evidence_page: response.evidence_page,
    debug: {
      original_query: response.original_query,
      parser_used: response.parser_used,
      llm_attempted: response.llm_attempted,
      llm_accepted: response.llm_accepted,
      llm_rejection_reason: response.llm_rejection_reason,
      llm_raw_structured_output: response.llm_raw_structured_output,
      normalized_llm_output: response.normalized_llm_output,
      message_type: response.message_type,
      context_was_available: response.context_was_available,
      context_reference: response.context_reference,
      filters_before_context: response.filters_before_context,
      filters_after_context: response.filters_after_context,
      explicit_filters_detected: response.explicit_filters_detected,
      filters_before_validation: response.filters_before_validation,
      filters_after_validation: response.filters_after_validation,
      group_by: response.group_by,
      matching_vehicle_ids_count: response.matching_vehicle_ids_count,
      context_saved_vehicle_ids_count: response.context_saved_vehicle_ids_count,
      parsed_query: response.parsed_query,
      analytics_result: response.analytics_result,
      matching_vehicle_ids: response.matching_vehicle_ids,
      context_used: response.context_used,
    },
  };
}

function makeMessageId(role: ChatMessage["role"]) {
  return `${role}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function ChatEmptyState({ onPrompt }: { onPrompt: (event?: FormEvent, override?: string) => void }) {
  return (
    <section className="chat-empty-state">
      <div className="assistant-avatar" aria-hidden="true"><span>AI</span></div>
      <div>
        <h2>Ask questions about this video</h2>
        <p>You can count vehicles, inspect colours, compare classes, and open evidence from matching tracks.</p>
        <div className="suggestion-grid">
          {PROMPT_SUGGESTIONS.map((prompt) => (
            <button key={prompt} type="button" onClick={() => onPrompt(undefined, prompt)}>
              {prompt}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}

function ChatMessageCard({
  message,
  onShowEvidence,
  onShowMore,
  onSelectTrack,
  selectedTrack,
}: {
  message: ChatMessage;
  onShowEvidence: () => void;
  onShowMore: () => void;
  onSelectTrack: (track: TrackSelection) => void;
  selectedTrack: TrackSelection | null;
}) {
  const hasEvidence = Boolean(message.evidence?.length);
  const isAnalytics = message.debug?.parsed_query.intent !== "GENERAL_CHAT";
  const canShowEvidence = message.role === "assistant" && isAnalytics && Boolean(message.debug?.matching_vehicle_ids.length) && !hasEvidence;

  return (
    <article className={`chat-message-row ${message.role}`}>
      {message.role === "assistant" ? <div className="assistant-avatar" aria-hidden="true"><span>AI</span></div> : null}
      <div className="chat-message">
        <div className="chat-message-meta">
          <span>{message.role === "assistant" ? "Analytics Assistant" : "You"}</span>
          <time dateTime={message.timestamp}>{formatMessageTime(message.timestamp)}</time>
        </div>
        {message.role === "assistant" ? <ResultPresentation message={message} /> : <p>{message.text}</p>}
        {canShowEvidence ? (
          <button type="button" className="secondary-button compact-action" onClick={onShowEvidence}>
            Show evidence
          </button>
        ) : null}
        {hasEvidence ? (
          <EvidenceSection
            evidence={message.evidence ?? []}
            page={message.evidence_page}
            onShowMore={onShowMore}
            onSelectTrack={onSelectTrack}
            selectedTrack={selectedTrack}
          />
        ) : null}
        {message.debug ? <DebugDetails debug={message.debug} /> : null}
      </div>
    </article>
  );
}

function ResultPresentation({ message }: { message: ChatMessage }) {
  const intent = message.debug?.parsed_query.intent?.toUpperCase();
  const analytics = message.debug?.analytics_result ?? {};
  const metrics = getResultMetrics(intent, analytics);
  const intervals = getIntervalResults(analytics);
  const groups = getGroupRows(analytics);

  return (
    <div className="result-block">
      <p>{message.text}</p>
      {metrics.length ? (
        <div className="result-metric-grid" aria-label="Result metrics">
          {metrics.map((metric) => (
            <div key={metric.label} className="result-metric">
              <strong>{metric.value}</strong>
              <span>{metric.label}</span>
            </div>
          ))}
        </div>
      ) : null}
      {groups.length ? (
        <div className="result-list" aria-label="Result breakdown">
          {groups.map((row) => (
            <div key={row.label}>
              <span>{row.label}</span>
              <strong>{row.value}</strong>
            </div>
          ))}
        </div>
      ) : null}
      {intervals.length ? (
        <div className="interval-list" aria-label="Time interval results">
          {intervals.map((interval) => (
            <div key={interval.label} className="interval-card">
              <strong>{interval.label}</strong>
              {interval.detail ? <span>{interval.detail}</span> : null}
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function EvidenceSection({
  evidence,
  page,
  onShowMore,
  onSelectTrack,
  selectedTrack,
}: {
  evidence: VehicleEvidence[];
  page?: EvidencePage;
  onShowMore: () => void;
  onSelectTrack: (track: TrackSelection) => void;
  selectedTrack: TrackSelection | null;
}) {
  const total = page?.matching_total ?? evidence.length;
  const shown = page?.shown_count ?? evidence.length;
  const remaining = page?.evidence_remaining_count ?? 0;
  return (
    <section className="evidence-section" aria-label="Matching evidence">
      <div className="evidence-section-heading">
        <div>
          <h3>Evidence</h3>
          <p>Showing {Math.min(shown, total)} of {total} vehicles</p>
        </div>
        {remaining > 0 ? (
          <button type="button" className="secondary-button compact-action" onClick={onShowMore}>
            Show more
          </button>
        ) : (
          <span>All {total} matching vehicles shown.</span>
        )}
      </div>
      <EvidenceGrid evidence={evidence} onSelectTrack={onSelectTrack} selectedTrack={selectedTrack} />
    </section>
  );
}

function EvidenceGrid({
  evidence,
  onSelectTrack,
  selectedTrack,
}: {
  evidence: VehicleEvidence[];
  onSelectTrack?: (track: TrackSelection) => void;
  selectedTrack?: TrackSelection | null;
}) {
  return (
    <div className="chat-evidence-grid">
      {evidence.map((item) => {
        const runId = runIdFromEvidence(item);
        const isSelected =
          selectedTrack?.cameraId === item.camera_id &&
          selectedTrack?.trackId === item.track_id &&
          (selectedTrack?.runId ?? null) === (runId ?? null);
        return (
          <article key={item.vehicle_id} className={`chat-evidence-card ${isSelected ? "selected" : ""}`}>
            <span className="evidence-accent" aria-hidden="true" />
            {item.image_url ? (
              <img src={item.image_url} alt={`${item.vehicle_class} ${item.colour} crop for ${item.track_id}`} loading="lazy" />
            ) : (
              <div className="thumb-placeholder">No crop</div>
            )}
            <div className="evidence-card-body">
              <strong>{item.track_id}</strong>
              <div className="evidence-badge-row">
                <span>{item.vehicle_class}</span>
                <span className={`vehicle-colour-badge colour-${item.colour.toLowerCase()}`}>{item.colour}</span>
              </div>
              <span className="evidence-seen">Seen {formatVideoTime(item.first_seen_seconds)} - {formatVideoTime(item.last_seen_seconds)}</span>
              {onSelectTrack ? (
                <button
                  type="button"
                  className="evidence-track-link"
                  aria-pressed={isSelected}
                  onClick={() => onSelectTrack({ runId, cameraId: item.camera_id, trackId: item.track_id })}
                >
                  {isSelected ? "Selected" : "View Track"}
                </button>
              ) : null}
            </div>
          </article>
        );
      })}
    </div>
  );
}

function DebugDetails({ debug }: { debug: NonNullable<ChatMessage["debug"]> }) {
  return (
    <details className="chat-debug">
      <summary>Developer Details</summary>
      <pre>{JSON.stringify(debug, null, 2)}</pre>
    </details>
  );
}

function TypingState() {
  return (
    <div className="chat-message-row assistant">
      <div className="assistant-avatar is-thinking" aria-hidden="true"><span>AI</span></div>
      <div className="chat-message loading">
        <span>Analytics Assistant is analyzing</span>
        <span className="typing-dots" aria-hidden="true"><i /><i /><i /></span>
      </div>
    </div>
  );
}

function ChatComposer({
  input,
  isSending,
  onInput,
  onSubmit,
}: {
  input: string;
  isSending: boolean;
  onInput: (value: string) => void;
  onSubmit: (event?: FormEvent, override?: string) => void;
}) {
  return (
    <div className="chat-composer-shell">
      <div className="prompt-chip-row" aria-label="Example prompts">
        {PROMPT_SUGGESTIONS.map((prompt) => (
          <button key={prompt} type="button" onClick={() => onSubmit(undefined, prompt)} disabled={isSending}>
            {prompt}
          </button>
        ))}
      </div>
      <form className="chat-input-row" onSubmit={onSubmit}>
        <input
          value={input}
          onChange={(event) => onInput(event.target.value)}
          placeholder="Ask about vehicles, colours, time ranges, or evidence..."
          aria-label="Video chat message"
        />
        <button type="submit" disabled={isSending || !input.trim()}>
          {isSending ? "Sending..." : "Send"}
        </button>
      </form>
    </div>
  );
}

function VideoContextPanel({
  selectedRun,
  runId,
  latestMessage,
  selectedTrack,
}: {
  selectedRun?: RunSummary;
  runId: string;
  latestMessage?: ChatMessage;
  selectedTrack: TrackSelection | null;
}) {
  const filters = latestMessage?.debug?.parsed_query
    ? [
        ...latestMessage.debug.parsed_query.include_classes,
        ...latestMessage.debug.parsed_query.include_colours,
        latestMessage.debug.parsed_query.camera_id,
      ].filter(Boolean)
    : [];
  const matches = latestMessage?.debug?.matching_vehicle_ids.length ?? latestMessage?.evidence_page?.matching_total ?? 0;

  return (
    <aside className="video-context-panel" aria-label="Video chat context">
      {selectedTrack ? (
        <section className="selected-track-panel" aria-label="Selected track details">
          <TrackDetailPanel selection={selectedTrack} compact showFullPageAction />
        </section>
      ) : (
        <>
          <section>
            <h2>Current Video</h2>
            <dl className="context-list">
              <div><dt>Run</dt><dd>{selectedRun?.run_id ?? runId}</dd></div>
              <div><dt>Status</dt><dd>{selectedRun?.status ?? "Latest"}</dd></div>
              <div><dt>Duration</dt><dd>{selectedRun?.duration_seconds != null ? formatVideoTime(selectedRun.duration_seconds) : "Unavailable"}</dd></div>
              <div><dt>Completed vehicles</dt><dd>{selectedRun?.track_count ?? "Unavailable"}</dd></div>
            </dl>
          </section>
          <section>
            <h2>Latest Result</h2>
            {latestMessage ? (
              <div className="latest-result-card">
                <span>Query</span>
                <strong>{latestMessage.debug?.original_query ?? latestMessage.text}</strong>
                <div className="context-stat-row">
                  <div><span>Matches</span><strong>{matches}</strong></div>
                  <div><span>Parser</span><strong>{latestMessage.debug?.parser_used ?? "n/a"}</strong></div>
                </div>
                {filters.length ? (
                  <div className="filter-chip-row">
                    {filters.map((filter) => <span key={String(filter)}>{filter}</span>)}
                  </div>
                ) : null}
              </div>
            ) : (
              <div className="empty-inline">No result yet.</div>
            )}
          </section>
          <section>
            <h2>Track Inspector</h2>
            <div className="empty-inline">Select a vehicle from the chat results to inspect its track.</div>
          </section>
        </>
      )}
    </aside>
  );
}

function runIdFromEvidence(item: VehicleEvidence): string | null {
  try {
    const url = new URL(item.track_detail_url, "http://local");
    return url.searchParams.get("run_id");
  } catch {
    return null;
  }
}

function formatMessageTime(timestamp: string) {
  const value = new Date(timestamp);
  if (Number.isNaN(value.getTime())) return "";
  return value.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function getResultMetrics(intent: string | undefined, analytics: Record<string, unknown>) {
  if (!intent || !["COUNT", "COMPARE", "GROUP", "SUMMARY"].includes(intent)) return [];
  const labels = [
    ["total", "Total"],
    ["count", "Matches"],
    ["matching_count", "Matches"],
    ["total_unique_vehicles", "Total vehicles"],
    ["difference", "Difference"],
  ] as const;
  return labels
    .map(([key, label]) => ({ label, value: analytics[key] }))
    .filter((metric) => typeof metric.value === "number" || typeof metric.value === "string")
    .map((metric) => ({ label: metric.label, value: String(metric.value) }));
}

function getGroupRows(analytics: Record<string, unknown>) {
  const source = analytics.by_type ?? analytics.by_class ?? analytics.by_colour ?? analytics.counts;
  if (!source || typeof source !== "object" || Array.isArray(source)) return [];
  return Object.entries(source as Record<string, unknown>)
    .filter(([, value]) => typeof value === "number" || typeof value === "string")
    .slice(0, 8)
    .map(([label, value]) => ({ label: formatLabel(label), value: String(value) }));
}

function getIntervalResults(analytics: Record<string, unknown>) {
  const intervals = analytics.intervals ?? analytics.matching_intervals ?? analytics.time_intervals;
  if (!Array.isArray(intervals)) return [];
  return intervals.slice(0, 4).map((interval, index) => {
    if (typeof interval === "string") return { label: interval, detail: "" };
    if (!interval || typeof interval !== "object") return { label: `Interval ${index + 1}`, detail: "" };
    const row = interval as Record<string, unknown>;
    const start = typeof row.start_time === "number" ? formatVideoTime(row.start_time) : row.start ?? row.start_time ?? "";
    const end = typeof row.end_time === "number" ? formatVideoTime(row.end_time) : row.end ?? row.end_time ?? "";
    const detail = Object.entries(row)
      .filter(([key]) => !["start", "end", "start_time", "end_time"].includes(key))
      .filter(([, value]) => typeof value === "number" || typeof value === "string")
      .map(([key, value]) => `${formatLabel(key)} ${value}`)
      .join(" · ");
    return { label: start || end ? `${start}-${end}` : `Interval ${index + 1}`, detail };
  });
}

function formatLabel(value: string) {
  return value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function describeChatError(error: unknown): string {
  if (error instanceof ApiError) {
    const detail = error.detail as { detail?: unknown } | null;
    const payload = typeof detail?.detail === "object" && detail.detail !== null ? detail.detail as { detail?: string; error?: string } : null;
    if (error.status === 400) {
      return payload?.detail || "I could not understand that question.";
    }
    if (error.status === 404) {
      return "Run not found.";
    }
    if (error.status >= 500) {
      return "Video chat is temporarily unavailable.";
    }
  }
  return error instanceof Error ? error.message : "Video chat request failed.";
}
