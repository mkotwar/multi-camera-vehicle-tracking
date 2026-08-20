import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { ApiError } from "../api/client";
import { fetchRuns } from "../api/runs";
import { sendVideoChatMessage } from "../api/videoChat";
import { TrackDetailPanel, type TrackSelection } from "../components/track/TrackDetailPanel";
import type { RunSummary } from "../types/run";
import type { ChatMessage, ChatVehicleQuery, EvidencePage, VideoChatResponse, VehicleEvidence } from "../types/videoChat";
import {
  createVideoChatSession,
  getOrCreateVideoChatSessionForRunIds,
  loadActiveVideoChatRunIds,
  loadActiveVideoChatRunId,
  replaceVideoChatSessionForRunIds,
  saveActiveVideoChatRunIds,
  saveVideoChatSessionForRun,
} from "../utils/videoChatPersistence";
import { resolvePlatePresentation } from "../utils/plates";
import { formatVideoTime } from "../utils/time";

const PROMPT_SUGGESTIONS = [
  "How many cars were there?",
  "Show white cars",
  "When were bikes more than cars?",
  "What colours were the motorcycles?",
];

export function VideoChatPage() {
  const initialRunIds = loadActiveVideoChatRunIds() ?? [loadActiveVideoChatRunId() ?? "latest"];
  const [chatSession, setChatSession] = useState(() => (initialRunIds.length ? getOrCreateVideoChatSessionForRunIds(initialRunIds) : createVideoChatSession("latest", [])));
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [runIds, setRunIds] = useState<string[]>(sessionRunIds(chatSession));
  const [messages, setMessages] = useState<ChatMessage[]>(chatSession.messages);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isRunMenuOpen, setIsRunMenuOpen] = useState(false);
  const [runQuery, setRunQuery] = useState("");
  const [userIsReading, setUserIsReading] = useState(false);
  const [selectedTrack, setSelectedTrack] = useState<TrackSelection | null>(null);
  const chatHistoryRef = useRef<HTMLDivElement | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);
  const runMenuRef = useRef<HTMLDivElement | null>(null);
  const runIdsRef = useRef(runIds);
  const chatSessionRef = useRef(chatSession);

  useEffect(() => {
    runIdsRef.current = runIds;
  }, [runIds]);

  useEffect(() => {
    chatSessionRef.current = chatSession;
  }, [chatSession]);

  useEffect(() => {
    if (!isRunMenuOpen) return;
    const handlePointerDown = (event: MouseEvent) => {
      if (!runMenuRef.current?.contains(event.target as Node)) {
        setIsRunMenuOpen(false);
      }
    };
    window.addEventListener("mousedown", handlePointerDown);
    return () => window.removeEventListener("mousedown", handlePointerDown);
  }, [isRunMenuOpen]);

  useEffect(() => {
    void fetchRuns()
      .then((items) => {
        setRuns(items);
        if (items[0]?.run_id && messages.length === 0 && runIds.includes("latest")) {
          loadRunSession([items[0].run_id]);
        }
      })
      .catch(() => setRuns([]));
  }, []);

  useEffect(() => {
    if (!userIsReading) {
      endRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
    }
  }, [messages, isSending, userIsReading]);

  const selectedRuns = useMemo(() => runIds.map((runId) => runs.find((item) => item.run_id === runId)).filter(Boolean) as RunSummary[], [runIds, runs]);
  const selectedRun = selectedRuns.length === 1 ? selectedRuns[0] : undefined;
  const latestAssistantMessage = useMemo(() => [...messages].reverse().find((message) => message.role === "assistant"), [messages]);
  const camerasInScope = selectedRuns.reduce((total, run) => total + Number(run.camera_count ?? 0), 0);
  const filteredRuns = useMemo(() => {
    const query = runQuery.trim().toLowerCase();
    if (!query) return runs;
    return runs.filter((run) => `${run.run_id} ${run.status ?? ""}`.toLowerCase().includes(query));
  }, [runQuery, runs]);

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
    if (runIds.length === 0) {
      setError("Select at least one run to start the chat.");
      return;
    }
    const currentRunIds = runIdsRef.current;
    const sessionId = chatSessionRef.current.session_id;
    const userMessage: ChatMessage = { id: makeMessageId("user"), role: "user", text, timestamp: new Date().toISOString() };
    persistMessages((current) => [...current, userMessage]);
    setInput("");
    setError(null);
    setIsSending(true);
    try {
      const response = await sendVideoChatMessage({ message: text, run_id: currentRunIds[0], run_ids: currentRunIds, session_id: sessionId });
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

  const startNewChat = (nextRunIds = runIdsRef.current) => {
    const nextSession = replaceVideoChatSessionForRunIds(nextRunIds);
    setChatSession(nextSession);
    setRunIds(nextRunIds);
    setMessages([]);
    setSelectedTrack(null);
    setInput("");
    setError(null);
    setIsRunMenuOpen(false);
    setRunQuery("");
    setUserIsReading(false);
  };

  const changeRuns = (nextRunIds: string[]) => {
    const normalized = normalizeRunIds(nextRunIds, runs);
    if (normalized.length === 0) {
      clearRunSelection();
      return;
    }
    loadRunSession(normalized);
  };

  function loadRunSession(nextRunIds: string[]) {
    const cleaned = normalizeRunIds(nextRunIds, runs);
    const nextSession = getOrCreateVideoChatSessionForRunIds(cleaned);
    setChatSession(nextSession);
    setRunIds(sessionRunIds(nextSession));
    setMessages(nextSession.messages);
    setSelectedTrack(null);
    setError(null);
    setIsRunMenuOpen(false);
    setRunQuery("");
    setUserIsReading(false);
    saveActiveVideoChatRunIds(cleaned);
    if (nextSession.messages.length === 0) {
      saveVideoChatSessionForRun(nextSession);
    }
  }

  function clearRunSelection() {
    const nextSession = createVideoChatSession("latest", []);
    setChatSession(nextSession);
    setRunIds([]);
    setMessages([]);
    setSelectedTrack(null);
    setInput("");
    setError(null);
    setIsRunMenuOpen(false);
    setRunQuery("");
    setUserIsReading(false);
    saveActiveVideoChatRunIds([]);
  }

  function persistMessages(updater: (current: ChatMessage[]) => ChatMessage[]) {
    setMessages((current) => {
      const nextMessages = updater(current);
      const currentRunIds = runIdsRef.current;
      const currentSession = chatSessionRef.current;
      const nextSession = { ...currentSession, run_id: currentRunIds[0] ?? currentSession.run_id, run_ids: currentRunIds, messages: nextMessages };
      setChatSession(nextSession);
      saveVideoChatSessionForRun(nextSession);
      return nextMessages;
    });
  }

  const runBadges = [
    { label: "Runs", value: selectedRuns.length > 1 ? `${selectedRuns.length} selected` : selectedRun?.run_id ?? runIds[0] ?? "None selected" },
    { label: "Cameras", value: selectedRuns.length ? camerasInScope : "None selected" },
    { label: "Duration", value: selectedRun?.duration_seconds != null ? formatVideoTime(selectedRun.duration_seconds) : selectedRuns.length > 1 ? "Multiple runs" : "Saved run" },
    { label: "Vehicles", value: selectedRuns.length ? selectedRuns.reduce((total, run) => total + Number(run.physical_vehicle_count ?? run.track_count ?? 0), 0) : "Unavailable" },
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
            <div key={badge.label} className={`run-badge ${badge.label === "Runs" ? "active" : ""}`}>
              <span>{badge.label}</span>
              <strong>{badge.value}</strong>
            </div>
          ))}
        </div>
      </section>

      <section className="video-chat-workspace">
        <section className="chat-panel" aria-label="Video analytics assistant">
          <div className="chat-panel-toolbar">
            <div ref={runMenuRef} className="run-selector-shell" aria-label="Video chat run">
              <div className="run-selector-heading">
                <span>Runs / videos</span>
                <small className="run-selector-summary">
                  {runIds.length === 0 ? "No runs selected" : `${runIds.length} run${runIds.length === 1 ? "" : "s"} selected`}
                </small>
              </div>
              <button
                type="button"
                className={`run-selector-trigger ${isRunMenuOpen ? "open" : ""}`}
                aria-label="Select runs"
                aria-expanded={isRunMenuOpen}
                aria-haspopup="dialog"
                aria-controls="video-chat-run-menu"
                onClick={() => setIsRunMenuOpen((current) => !current)}
              >
                <strong>{runIds.length === 0 ? "Choose runs" : runIds.length === 1 ? runIds[0] : `${runIds.length} runs selected`}</strong>
                <span>{runIds.length === 0 ? "Search and select one or more processed runs" : `${camerasInScope} cameras in scope`}</span>
              </button>
              <div className="run-selector-scope" aria-live="polite">
                <strong>Selected runs: {runIds.length}</strong>
                <span>Cameras in scope: {runIds.length === 0 ? 0 : camerasInScope}</span>
              </div>
              {isRunMenuOpen ? (
                <div id="video-chat-run-menu" className="run-selector-popover" role="dialog" aria-label="Run selection menu">
                  <label className="run-selector-search">
                    <span>Search runs</span>
                    <input
                      value={runQuery}
                      onChange={(event) => setRunQuery(event.target.value)}
                      placeholder="Search by run ID or status"
                      aria-label="Search runs"
                    />
                  </label>
                  <div className="run-selector-actions">
                    <button type="button" className="secondary-button compact-action" onClick={() => changeRuns(runs.map((run) => run.run_id))} disabled={runs.length === 0}>
                      Select all
                    </button>
                    <button type="button" className="secondary-button compact-action" onClick={() => changeRuns([])} disabled={runIds.length === 0}>
                      Clear
                    </button>
                  </div>
                  <div className="run-selector-list" role="group" aria-label="Video chat run">
                    {runs.length === 0 ? (
                      <label className="run-selector-option">
                        <input type="checkbox" checked readOnly />
                        <span>Latest</span>
                      </label>
                    ) : filteredRuns.length === 0 ? (
                      <div className="run-selector-empty">No runs match that search.</div>
                    ) : (
                      filteredRuns.map((run) => {
                        const checked = runIds.includes(run.run_id);
                        return (
                          <label key={run.run_id} className="run-selector-option">
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={() => {
                                const nextRunIds = checked ? runIds.filter((runId) => runId !== run.run_id) : [...runIds, run.run_id];
                                changeRuns(nextRunIds);
                              }}
                            />
                            <span>{run.run_id}</span>
                            <small>{run.status ?? "Saved run"}</small>
                          </label>
                        );
                      })
                    )}
                  </div>
                </div>
              ) : null}
            </div>
            <span className="status status-completed">{runIds.length === 0 ? "Choose runs" : selectedRuns.length > 1 ? "All selected runs" : selectedRun?.status ?? "Latest"}</span>
            <button type="button" className="secondary-button compact-action" onClick={() => startNewChat()}>
              New Chat
            </button>
          </div>

          <div ref={chatHistoryRef} className="chat-history" aria-label="Video analytics chat history" onScroll={handleScroll}>
            {messages.length === 0 ? <ChatEmptyState onPrompt={submitMessage} selectedRunCount={runIds.length} /> : null}
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
          runId={runIds[0] ?? "latest"}
          selectedRunIds={runIds}
          selectedRuns={selectedRuns}
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
      normalized_plan: response.normalized_plan,
      semantic_repair_applied: response.semantic_repair_applied,
      semantic_repair_notes: response.semantic_repair_notes,
      fallback_reason: response.fallback_reason,
      parser_model: response.parser_model,
      total_parser_ms: response.total_parser_ms,
      qwen_request_ms: response.qwen_request_ms,
      normalize_ms: response.normalize_ms,
      repair_ms: response.repair_ms,
      validation_ms: response.validation_ms,
      ollama_metadata: response.ollama_metadata,
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

function ChatEmptyState({ onPrompt, selectedRunCount }: { onPrompt: (event?: FormEvent, override?: string) => void; selectedRunCount: number }) {
  return (
    <section className="chat-empty-state">
      <div className="assistant-avatar" aria-hidden="true"><span>AI</span></div>
      <div>
        <h2>{selectedRunCount > 1 ? "Ask questions across these runs" : "Ask questions about this run"}</h2>
        <p>You can count vehicles, compare cameras or runs, filter by plates, and open evidence from matching tracks.</p>
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
  const parsed = message.debug?.parsed_query;
  const analytics = message.debug?.analytics_result ?? {};
  const lead = getResultLead(message);
  const metrics = getResultMetrics(parsed, analytics);
  const primaryMetric = getPrimaryMetric(parsed, analytics);
  const intervals = getIntervalResults(analytics);
  const groups = getGroupRows(parsed, analytics);
  const groupedSummary = getGroupedSummary(parsed, analytics);
  const filterChips = getFilterChips(parsed);
  const listValues = getValueList(parsed, analytics);
  const plateRows = getPlateLookupRows(parsed, analytics);
  const plateSummary = getPlateLookupSummary(parsed, analytics);
  const isPlateLookup = parsed?.intent?.toUpperCase() === "PLATE_LOOKUP";

  return (
    <div className="result-block">
      {primaryMetric ? (
        <section className="result-primary-metric" aria-label="Primary result metric">
          {primaryMetric.title ? <span className="result-primary-eyebrow">{primaryMetric.title}</span> : null}
          <strong>{primaryMetric.value}</strong>
          <span>{primaryMetric.label}</span>
          {lead ? <p>{lead}</p> : null}
        </section>
      ) : (
        <p>{lead}</p>
      )}
      {filterChips.length ? (
        <div className="result-chip-row" aria-label="Applied filters">
          {filterChips.map((chip) => (
            <span key={chip}>{chip}</span>
          ))}
        </div>
      ) : null}
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
      {groupedSummary.length ? (
        <div className="grouped-summary-grid" aria-label="Structured summary">
          {groupedSummary.map((group) => (
            <section key={group.key} className="grouped-summary-card">
              <div className="grouped-summary-header">
                <strong>{group.title}</strong>
                {group.subtitle ? <span>{group.subtitle}</span> : null}
              </div>
              <div className="grouped-summary-stats">
                <div>
                  <span>Vehicles</span>
                  <strong>{group.total}</strong>
                </div>
                {group.window ? (
                  <div>
                    <span>Seen</span>
                    <strong>{group.window}</strong>
                  </div>
                ) : null}
              </div>
              {group.highlights.length ? (
                <div className="result-chip-row" aria-label={`Highlights for ${group.title}`}>
                  {group.highlights.map((item) => (
                    <span key={item}>{item}</span>
                  ))}
                </div>
              ) : null}
            </section>
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
      {listValues.length ? (
        <div className="result-chip-row" aria-label="Result values">
          {listValues.map((value) => (
            <span key={value}>{value}</span>
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
      {isPlateLookup && plateSummary ? (
        <section className="plate-lookup-summary" aria-label="Plate lookup summary">
          <div className="plate-lookup-summary__header">
            <strong>{plateSummary.title}</strong>
            {plateSummary.subtitle ? <span>{plateSummary.subtitle}</span> : null}
          </div>
          {plateSummary.chips.length ? (
            <div className="result-chip-row" aria-label="Plate lookup details">
              {plateSummary.chips.map((chip) => (
                <span key={chip}>{chip}</span>
              ))}
            </div>
          ) : null}
        </section>
      ) : null}
      {plateRows.length ? (
        <div className="plate-lookup-list" aria-label="Number plate results">
          {plateRows.map((row) => {
            const plate = resolvePlatePresentation({
              plateText: row.plate_text,
              plateDetected: row.plate_detected,
              readableMissingLabel: "No plate detected",
              unreadableLabel: "Plate detected, unreadable",
            });
            return (
              <article key={`${row.run_id ?? "run"}-${row.vehicle_id}`} className="plate-lookup-card">
                <div className="plate-lookup-card__meta">
                  <strong>{row.vehicle_id}</strong>
                  <span>{row.run_id ? `Run ${row.run_id}` : "Run unavailable"}</span>
                </div>
                <span className={`plate-badge ${plate.state}`}>{plate.label}</span>
                <div className="plate-lookup-card__detail">
                  <span>{row.camera_id ? `Camera ${row.camera_id}` : "Camera unavailable"}</span>
                  <span>{row.track_id ? `Track ${row.track_id}` : "Track unavailable"}</span>
                </div>
              </article>
            );
          })}
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
        const imageUrl = item.best_crop_url ?? item.image_url;
        const memberTracks = item.member_track_ids ?? [];
        const displayTrack = memberTracks.length > 1 ? `${memberTracks.length} tracklets` : item.track_id;
        const plate = resolvePlatePresentation({
          plateText: item.plate_text,
          plateDetected: item.plate_detected ?? item.plate_readable,
          readableMissingLabel: "No plate detected",
          unreadableLabel: "Plate detected, unreadable",
        });
        const isSelected =
          selectedTrack?.cameraId === item.camera_id &&
          selectedTrack?.trackId === item.track_id &&
          (selectedTrack?.runId ?? null) === (runId ?? null);
        return (
          <article key={item.vehicle_id} className={`chat-evidence-card ${isSelected ? "selected" : ""}`}>
            <span className="evidence-accent" aria-hidden="true" />
            {imageUrl ? (
              <img src={imageUrl} alt={`${item.vehicle_class} ${item.colour} crop for ${item.vehicle_id}`} loading="lazy" />
            ) : (
              <div className="thumb-placeholder">No crop</div>
            )}
            <div className="evidence-card-body">
              <strong>{item.vehicle_id}</strong>
              <span className="evidence-seen">Run: {item.run_id ?? runId ?? "Unknown"}</span>
              <span className="evidence-seen">Camera: {item.camera_id}</span>
              <div className="evidence-badge-row">
                <span>{item.vehicle_class}</span>
                <span className={`vehicle-colour-badge colour-${item.colour.toLowerCase()}`}>{item.colour}</span>
              </div>
              <span className="evidence-seen">Track: {displayTrack}</span>
              <div className="evidence-plate-block">
                <span className="evidence-seen evidence-seen--label">Number plate</span>
                <span className={`plate-badge ${plate.state}`}>{plate.label}</span>
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
          placeholder="Ask about selected runs, cameras, plates, time ranges, or evidence..."
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
  selectedRunIds,
  selectedRuns,
  latestMessage,
  selectedTrack,
}: {
  selectedRun?: RunSummary;
  runId: string;
  selectedRunIds: string[];
  selectedRuns: RunSummary[];
  latestMessage?: ChatMessage;
  selectedTrack: TrackSelection | null;
}) {
  const filters = latestMessage?.debug?.parsed_query
    ? [
        ...latestMessage.debug.parsed_query.include_classes,
        ...latestMessage.debug.parsed_query.include_colours,
        latestMessage.debug.parsed_query.camera_id,
        ...(latestMessage.debug.parsed_query.include_camera_ids ?? []),
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
            <h2>Current Scope</h2>
            <dl className="context-list">
              <div><dt>Run</dt><dd>{selectedRunIds.length > 1 ? `${selectedRunIds.length} selected` : selectedRun?.run_id ?? runId}</dd></div>
              <div><dt>Scope</dt><dd>{selectedRunIds.length === 0 ? "No runs selected" : selectedRunIds.length > 1 ? "All selected runs" : selectedRun?.run_id ?? runId}</dd></div>
              <div><dt>Status</dt><dd>{selectedRunIds.length === 0 ? "Choose runs" : selectedRunIds.length > 1 ? `${selectedRunIds.length} runs` : selectedRun?.status ?? "Latest"}</dd></div>
              <div><dt>Duration</dt><dd>{selectedRun?.duration_seconds != null ? formatVideoTime(selectedRun.duration_seconds) : selectedRunIds.length > 1 ? "Multiple runs" : "Unavailable"}</dd></div>
              <div><dt>Completed vehicles</dt><dd>{selectedRuns.length ? selectedRuns.reduce((total, run) => total + Number(run.physical_vehicle_count ?? run.track_count ?? 0), 0) : selectedRun?.physical_vehicle_count ?? selectedRun?.track_count ?? "Unavailable"}</dd></div>
              {selectedRun?.raw_track_count != null && selectedRun.raw_track_count !== (selectedRun.physical_vehicle_count ?? selectedRun.track_count) ? (
                <div><dt>Raw tracks</dt><dd>{selectedRun.raw_track_count}</dd></div>
              ) : null}
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
  if (item.run_id) return item.run_id;
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

function getResultLead(message: ChatMessage) {
  const parsed = message.debug?.parsed_query;
  const analytics = message.debug?.analytics_result ?? {};
  const intent = parsed?.intent?.toUpperCase();
  const ranking = getRankingSummary(parsed, analytics);
  if (ranking?.lead) {
    return ranking.lead;
  }
  if (intent === "SUMMARY" && parsed?.group_by) {
    return `Traffic summary by ${formatGroupBy(parsed.group_by)}.`;
  }
  if (intent === "GROUP" && parsed?.group_by) {
    const total = readNumericResult(analytics.total ?? analytics.matching_count ?? analytics.count);
    return total == null ? message.text : `I found ${total} matching vehicles.`;
  }
  return message.text;
}

function getResultMetrics(parsed: ChatVehicleQuery | undefined, analytics: Record<string, unknown>) {
  const intent = parsed?.intent?.toUpperCase();
  const subject = parsed?.subject;
  const groupBy = parsed?.group_by;
  if (!intent) return [];
  if (intent === "COUNT") {
    const total = readNumericResult(analytics.total ?? analytics.count ?? analytics.matching_count);
    if (total == null) return [];
    return [{ label: subject === "runs" ? "Runs" : "Matching vehicles", value: String(total) }];
  }
  if (intent === "SUMMARY" && !groupBy) {
    const total = readNumericResult(analytics.total_unique_vehicles);
    return total == null ? [] : [{ label: "Completed vehicles", value: String(total) }];
  }
  if (intent === "GROUP") {
    const total = readNumericResult(analytics.total);
    return total == null ? [] : [{ label: "Matching vehicles", value: String(total) }];
  }
  if (intent === "COMPARE") {
    const metrics = [
      analytics.left && analytics.left_total != null ? { label: String(analytics.left), value: String(analytics.left_total) } : null,
      analytics.right && analytics.right_total != null ? { label: String(analytics.right), value: String(analytics.right_total) } : null,
      analytics.answer ? { label: "Result", value: String(analytics.answer) } : null,
    ];
    return metrics.filter(Boolean) as Array<{ label: string; value: string }>;
  }
  if (intent === "PLATE_LOOKUP") {
    const total = readNumericResult(analytics.target_total ?? analytics.total);
    return total == null ? [] : [{ label: total === 1 ? "Vehicle inspected" : "Vehicles inspected", value: String(total) }];
  }
  return [];
}

function getPrimaryMetric(parsed: ChatVehicleQuery | undefined, analytics: Record<string, unknown>) {
  const intent = parsed?.intent?.toUpperCase();
  const ranking = getRankingSummary(parsed, analytics);
  if (ranking) {
    return {
      title: ranking.title,
      value: ranking.value,
      label: ranking.label,
    };
  }
  const total = readNumericResult(
    analytics.total ??
    analytics.count ??
    analytics.matching_count ??
    analytics.total_unique_vehicles,
  );
  if (intent === "COUNT" && total != null) {
    return {
      title: getPrimaryMetricTitle(parsed),
      value: String(total),
      label: total === 1 ? (parsed?.subject === "runs" ? "run" : "vehicle") : (parsed?.subject === "runs" ? "runs" : "vehicles"),
    };
  }
  if (intent === "SUMMARY" && !parsed?.group_by && total != null) {
    return {
      title: getPrimaryMetricTitle(parsed),
      value: String(total),
      label: total === 1 ? "vehicle" : "vehicles",
    };
  }
  if (intent === "PLATE_LOOKUP" && total != null) {
    return {
      title: getPrimaryMetricTitle(parsed),
      value: String(total),
      label: total === 1 ? "plate lookup" : "plate lookups",
    };
  }
  return null;
}

function getPrimaryMetricTitle(parsed: ChatVehicleQuery | undefined) {
  if (!parsed) return "Result";
  if (parsed.subject === "runs") {
    return parsed.run_filter === "multiple_cameras" ? "Runs with multiple cameras" : "Selected runs";
  }
  const includeClasses = parsed.include_classes.map((value) => formatLabel(value));
  const includeColours = parsed.include_colours.map((value) => formatLabel(value));
  const titleParts = [...includeColours, ...includeClasses];
  if (titleParts.length) {
    return titleParts.join(" ");
  }
  if (parsed.plate_text) {
    return `Plate ${parsed.plate_text}`;
  }
  if (parsed.plate_readable === false) {
    return "Unreadable plates";
  }
  if (parsed.plate_presence === "detected") {
    return "Vehicles with number plates";
  }
  if (parsed.intent?.toUpperCase() === "PLATE_LOOKUP") {
    return parsed.context_resolution === "multiple" ? "Number plates" : "Number plate";
  }
  return "Result";
}

function getGroupRows(parsed: ChatVehicleQuery | undefined, analytics: Record<string, unknown>) {
  const intent = parsed?.intent?.toUpperCase();
  if (getRankingSummary(parsed, analytics)) return [];
  if (intent === "GROUP" && parsed?.sort_by && parsed.limit && parsed.limit > 1) {
    const ranking = analytics.ranking_result;
    if (ranking && typeof ranking === "object" && !Array.isArray(ranking)) {
      const entries = Array.isArray((ranking as Record<string, unknown>).entries)
        ? ((ranking as Record<string, unknown>).entries as Array<Record<string, unknown>>)
        : [];
      return entries.slice(0, parsed.limit).map((entry) => ({
        label: formatRankingWinner(entry, parsed.group_by),
        value: String(readNumericResult(entry.count) ?? 0),
      }));
    }
  }
  if (intent === "SUMMARY" && parsed?.group_by) return [];
  const source = resolveGroupSource(parsed, analytics);
  if (!source) return [];
  return Object.entries(source)
    .filter(([, value]) => readNumericResult(value) != null && Number(value) > 0)
    .slice(0, 12)
    .map(([label, value]) => ({ label: formatGroupLabel(label, parsed?.group_by), value: String(value) }));
}

function getGroupedSummary(parsed: ChatVehicleQuery | undefined, analytics: Record<string, unknown>) {
  const groupBy = parsed?.group_by;
  if (parsed?.intent?.toUpperCase() !== "SUMMARY" || !groupBy) return [];
  const groups = analytics.groups;
  if (!groups || typeof groups !== "object" || Array.isArray(groups)) return [];
  return Object.entries(groups as Record<string, unknown>)
    .map(([key, value]) => {
      if (!value || typeof value !== "object" || Array.isArray(value)) return null;
      const group = value as Record<string, unknown>;
      const classes = formatCountHighlights(group.vehicle_classes);
      const colours = formatCountHighlights(group.colours);
      const window = formatTimeWindow(group.first_seen_seconds, group.last_seen_seconds);
      const [runTitle, cameraTitle] = key.split(" / ");
      return {
        key,
        title: groupBy === "run_camera" ? (runTitle || key) : formatGroupLabel(key, groupBy),
        subtitle: groupBy === "run_camera" && cameraTitle ? cameraTitle : undefined,
        total: String(readNumericResult(group.total_unique_vehicles) ?? 0),
        window,
        highlights: [...classes, ...colours].slice(0, 6),
      };
    })
    .filter(Boolean) as Array<{ key: string; title: string; subtitle?: string; total: string; window?: string; highlights: string[] }>;
}

function getFilterChips(parsed: ChatVehicleQuery | undefined) {
  if (!parsed) return [];
  const chips = [
    ...parsed.include_classes.map((value) => formatLabel(value)),
    ...parsed.include_colours.map((value) => formatLabel(value)),
    parsed.plate_presence ? `Plates: ${formatLabel(parsed.plate_presence)}` : null,
    parsed.plate_detected === true ? "Plate detected" : null,
    parsed.plate_detected === false ? "No plate detected" : null,
    parsed.plate_readable === false ? "Unreadable plate" : null,
    parsed.plate_text ? `Plate: ${parsed.plate_text}` : null,
    parsed.camera_id ? `Camera: ${parsed.camera_id}` : null,
    ...(parsed.include_camera_ids ?? []).map((value) => `Camera: ${value}`),
    parsed.run_filter === "multiple_cameras" ? "Multiple cameras only" : null,
  ];
  return chips.filter(Boolean) as string[];
}

function getValueList(parsed: ChatVehicleQuery | undefined, analytics: Record<string, unknown>) {
  const intent = parsed?.intent?.toUpperCase();
  if (intent === "UNIQUE_CLASSES") {
    return toStringList(analytics.vehicle_classes_present);
  }
  if (intent === "UNIQUE_COLOURS") {
    return toStringList(analytics.colours_present);
  }
  if (intent === "SUMMARY" && !parsed?.group_by) {
    return [
      ...formatCountHighlights(analytics.vehicle_classes ?? analytics.by_type),
      ...formatCountHighlights(analytics.colours ?? analytics.by_colour),
    ].slice(0, 10);
  }
  return [];
}

function getPlateLookupRows(parsed: ChatVehicleQuery | undefined, analytics: Record<string, unknown>) {
  if (parsed?.intent?.toUpperCase() !== "PLATE_LOOKUP") return [];
  const rows = analytics.plate_rows;
  if (!Array.isArray(rows)) return [];
  return rows.filter((row): row is {
    vehicle_id: string;
    run_id?: string | null;
    camera_id?: string | null;
    track_id?: string | null;
    plate_text?: string | null;
    plate_detected?: boolean | null;
  } => Boolean(row && typeof row === "object" && "vehicle_id" in row));
}

function getPlateLookupSummary(parsed: ChatVehicleQuery | undefined, analytics: Record<string, unknown>) {
  if (parsed?.intent?.toUpperCase() !== "PLATE_LOOKUP") return null;
  if (analytics.ambiguous) {
    return {
      title: "Plate lookup needs a narrower target",
      subtitle: "The previous result contains multiple vehicles.",
      chips: toStringList(analytics.candidate_vehicle_ids).slice(0, 5).map((value) => `Candidate: ${value}`),
    };
  }
  const readable = readNumericResult(analytics.readable_count) ?? 0;
  const unreadable = readNumericResult(analytics.detected_unreadable_count) ?? 0;
  const missing = readNumericResult(analytics.no_plate_count) ?? 0;
  const chips = [
    readable ? `${readable} readable` : null,
    unreadable ? `${unreadable} unreadable` : null,
    missing ? `${missing} not detected` : null,
  ].filter(Boolean) as string[];
  return {
    title: parsed.context_resolution === "multiple" ? "Number plate results" : "Number plate result",
    subtitle: chips.length ? undefined : "No plate details are available for the selected result.",
    chips,
  };
}

function normalizeRunIds(runIds: string[], runs: RunSummary[]): string[] {
  const selected = new Set(runIds.map((item) => item.trim()).filter(Boolean));
  if (selected.size === 0) {
    return [];
  }
  const availableRunIds = runs.map((run) => run.run_id).filter((runId) => selected.has(runId));
  return availableRunIds.length > 0 ? availableRunIds : Array.from(selected);
}

function sessionRunIds(session: { run_id: string; run_ids?: string[] }): string[] {
  if (session.run_ids !== undefined) {
    return session.run_ids;
  }
  return session.run_id ? [session.run_id] : [];
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

function resolveGroupSource(parsed: ChatVehicleQuery | undefined, analytics: Record<string, unknown>) {
  const intent = parsed?.intent?.toUpperCase();
  const groupBy = parsed?.group_by;
  if (intent !== "GROUP") return null;
  const sourceKey = groupBy === "run_camera"
    ? "by_run_camera"
    : groupBy === "run"
      ? "by_run"
      : groupBy === "camera"
        ? "by_camera"
        : groupBy === "colour"
          ? "by_colour"
          : "by_class";
  const source = analytics[sourceKey];
  if (!source || typeof source !== "object" || Array.isArray(source)) return null;
  return source as Record<string, unknown>;
}

function getRankingSummary(parsed: ChatVehicleQuery | undefined, analytics: Record<string, unknown>) {
  if (parsed?.intent?.toUpperCase() !== "GROUP" || parsed.limit !== 1 || !parsed.sort_by) return null;
  const ranking = analytics.ranking_result;
  if (!ranking || typeof ranking !== "object" || Array.isArray(ranking)) return null;
  const winners = Array.isArray((ranking as Record<string, unknown>).winners)
    ? ((ranking as Record<string, unknown>).winners as Array<Record<string, unknown>>)
    : [];
  if (!winners.length) return null;
  const count = readNumericResult(winners[0]?.count);
  const groupBy = parsed.group_by;
  const noun = groupBy === "run"
    ? "run"
    : groupBy === "run_camera" || groupBy === "camera"
      ? "camera"
      : groupBy === "colour"
        ? "vehicle colour"
        : "vehicle class";
  const topLabel = parsed.sort_by === "count_asc" ? `Lowest ${noun}` : `Top ${noun}`;
  if (winners.length > 1) {
    const names = winners.map((item) => formatRankingWinner(item, groupBy)).join(", ");
    return {
      title: topLabel,
      value: `${winners.length} tied`,
      label: count == null ? "Tie" : `${count} ${count === 1 ? "vehicle" : "vehicles"} each`,
      lead: names,
    };
  }
  const winner = winners[0];
  const name = formatRankingWinner(winner, groupBy);
  const runId = typeof winner.run_id === "string" ? winner.run_id : null;
  return {
    title: topLabel,
    value: name,
    label: count == null ? "Vehicles" : `${count} ${count === 1 ? "vehicle" : "vehicles"}`,
    lead: groupBy === "run_camera" && runId ? `Run: ${runId}` : null,
  };
}

function formatRankingWinner(entry: Record<string, unknown>, groupBy?: string | null) {
  if (groupBy === "run_camera" && typeof entry.camera_id === "string" && entry.camera_id) {
    return entry.camera_id;
  }
  if (groupBy === "run" && typeof entry.run_id === "string" && entry.run_id) {
    return entry.run_id;
  }
  if (groupBy === "class" && typeof entry.label === "string") {
    return formatLabel(entry.label);
  }
  return typeof entry.label === "string" ? formatGroupLabel(entry.label, groupBy) : "Result";
}

function formatGroupBy(value: string) {
  return value === "run_camera" ? "run and camera" : formatLabel(value).toLowerCase();
}

function formatGroupLabel(value: string, groupBy?: string | null) {
  if (groupBy === "run_camera") {
    const [runId, cameraId] = value.split(" / ");
    if (runId && cameraId) {
      return `${runId} - ${cameraId}`;
    }
  }
  return formatLabel(value);
}

function readNumericResult(value: unknown) {
  return typeof value === "number" ? value : typeof value === "string" && value.trim() !== "" && !Number.isNaN(Number(value)) ? Number(value) : null;
}

function formatCountHighlights(value: unknown) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return [];
  return Object.entries(value as Record<string, unknown>)
    .filter(([, count]) => readNumericResult(count) != null && Number(count) > 0)
    .sort(([, left], [, right]) => Number(right) - Number(left))
    .slice(0, 3)
    .map(([label, count]) => `${formatLabel(label)} ${count}`);
}

function formatTimeWindow(firstSeen: unknown, lastSeen: unknown) {
  const start = typeof firstSeen === "number" ? formatVideoTime(firstSeen) : null;
  const end = typeof lastSeen === "number" ? formatVideoTime(lastSeen) : null;
  if (!start && !end) return undefined;
  return `${start ?? "?"} - ${end ?? "?"}`;
}

function toStringList(value: unknown) {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string" && item.trim().length > 0) : [];
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
