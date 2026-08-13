import { FormEvent, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useSearchParams } from "react-router-dom";
import { ApiError } from "../api/client";
import { fetchFilterOptions } from "../api/filters";
import { fetchTrackReconciliation } from "../api/runs";
import { fetchTracks } from "../api/tracks";
import { searchVehicles } from "../api/vehicleSearch";
import type { FilterOptions } from "../types/filters";
import type { ReconciliationAssociation, ReconciliationTrack, TrackReconciliationResult } from "../types/run";
import type { TrackRecord } from "../types/track";
import type { VehicleSearchResponse } from "../types/vehicleSearch";
import { formatVideoTime, parseVideoTime } from "../utils/time";

type Filters = {
  run_id: string;
  camera_id: string;
  vehicle_class: string;
  colour: string;
  track_id: string;
  from_time: string;
  to_time: string;
};

const DEFAULT_FILTERS: Filters = {
  run_id: "latest",
  camera_id: "",
  vehicle_class: "",
  colour: "",
  track_id: "",
  from_time: "",
  to_time: "",
};

const PAGE_SIZE = 25;
type TrackingViewMode = "raw" | "reconciled";

type ReconciledVehicleRow = {
  vehicle_id: string;
  camera_id: string;
  vehicle_class: string;
  colour: string;
  fragments: ReconciliationTrack[];
  accepted?: ReconciliationAssociation;
  recovered: boolean;
  first_seen_seconds: number | null;
  last_seen_seconds: number | null;
  gap_frames?: number | string | null;
  gap_seconds?: number | string | null;
  score?: number | string | null;
  second_best_score?: number | string | null;
  result: string;
};

function formatNumber(value: unknown): string {
  if (value === null || value === undefined || value === "") {
    return "0";
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toLocaleString() : String(value);
}

function formatScore(value: unknown): string {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(3) : "Unavailable";
}

function formatSeconds(value: unknown): string {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${numeric.toFixed(2)}s` : "Unavailable";
}

function shortTrackId(value: string): string {
  const parts = value.split(":");
  return parts[parts.length - 1] || value;
}

function trackColour(track: ReconciliationTrack): string {
  return String(track.vehicle_enrichment?.vehicle_colour?.label ?? "UNKNOWN").toUpperCase();
}

function trackClass(track: ReconciliationTrack): string {
  return String(track.vehicle_enrichment?.vehicle_class ?? track.final_class ?? "UNKNOWN").toUpperCase();
}

function buildReconciledRows(reconciliation: TrackReconciliationResult | null): ReconciledVehicleRow[] {
  if (!reconciliation?.available) return [];
  const completed = (reconciliation.tracks ?? []).filter((track) => String(track.status ?? "").toUpperCase() === "COMPLETED");
  const byVehicle = new Map<string, ReconciliationTrack[]>();
  for (const track of completed) {
    const vehicleId = String(track.vehicle_id || track.local_track_id);
    const group = byVehicle.get(vehicleId) ?? [];
    group.push(track);
    byVehicle.set(vehicleId, group);
  }
  const acceptedByVehicle = new Map<string, ReconciliationAssociation>();
  for (const association of reconciliation.accepted_associations ?? []) {
    acceptedByVehicle.set(association.vehicle_id, association);
  }
  return [...byVehicle.entries()]
    .map(([vehicleId, fragments]) => {
      const sortedFragments = [...fragments].sort((left, right) => Number(left.first_frame ?? 0) - Number(right.first_frame ?? 0));
      const accepted = acceptedByVehicle.get(vehicleId);
      const matchedFragment = sortedFragments.find((fragment) => fragment.reconciliation?.matched);
      const recovered = sortedFragments.length > 1 || Boolean(accepted || matchedFragment);
      return {
        vehicle_id: vehicleId,
        camera_id: String(sortedFragments[0]?.camera_id ?? ""),
        vehicle_class: trackClass(sortedFragments[0]),
        colour: trackColour(sortedFragments[0]),
        fragments: sortedFragments,
        accepted,
        recovered,
        first_seen_seconds: nullableNumber(sortedFragments[0]?.first_timestamp_seconds),
        last_seen_seconds: nullableNumber(sortedFragments[sortedFragments.length - 1]?.last_timestamp_seconds),
        gap_frames: accepted?.gap_frames ?? matchedFragment?.reconciliation?.time_gap_frames,
        gap_seconds: accepted?.gap_seconds ?? matchedFragment?.reconciliation?.time_gap_seconds,
        score: accepted?.score ?? matchedFragment?.reconciliation?.score,
        second_best_score: accepted?.second_best_score ?? matchedFragment?.reconciliation?.second_best_score,
        result: recovered ? "RECOVERED" : "ORIGINAL",
      };
    })
    .sort((left, right) => Number(left.fragments[0]?.first_frame ?? 0) - Number(right.fragments[0]?.first_frame ?? 0));
}

function nullableNumber(value: unknown): number | null {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function getActiveChips(filters: Filters): string[] {
  return [
    filters.run_id && filters.run_id !== "latest" ? `Run: ${filters.run_id}` : "",
    filters.camera_id ? `Camera: ${filters.camera_id}` : "",
    filters.vehicle_class ? `Class: ${filters.vehicle_class}` : "",
    filters.colour ? `Colour: ${filters.colour}` : "",
    filters.track_id ? `Track: ${filters.track_id}` : "",
    filters.from_time ? `From: ${filters.from_time}` : "",
    filters.to_time ? `To: ${filters.to_time}` : "",
  ].filter(Boolean);
}

export function VehicleSearchPage() {
  const [searchParams] = useSearchParams();
  const initialRunId = searchParams.get("run_id") ?? "latest";
  const [rows, setRows] = useState<TrackRecord[]>([]);
  const [filters, setFilters] = useState<Filters>({ ...DEFAULT_FILTERS, run_id: initialRunId });
  const [options, setOptions] = useState<FilterOptions>({ runs: ["latest"], cameras: [], vehicle_classes: [], colours: [] });
  const [currentPage, setCurrentPage] = useState(1);
  const [error, setError] = useState<string | null>(null);
  const [nlQuery, setNlQuery] = useState("");
  const [nlResult, setNlResult] = useState<VehicleSearchResponse | null>(null);
  const [nlError, setNlError] = useState<string | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [trackingView, setTrackingView] = useState<TrackingViewMode>("raw");
  const [reconciliation, setReconciliation] = useState<TrackReconciliationResult | null>(null);
  const [reconciliationError, setReconciliationError] = useState<string | null>(null);

  const load = async (next = filters) => {
    setError(null);
    const query = new URLSearchParams();
    if (next.run_id) query.set("run_id", next.run_id);
    if (next.camera_id) query.set("camera_id", next.camera_id);
    if (next.vehicle_class) query.set("vehicle_class", next.vehicle_class);
    if (next.colour) query.set("colour", next.colour);
    if (next.track_id) query.set("track_id", next.track_id);
    const fromSeconds = parseVideoTime(next.from_time);
    const toSeconds = parseVideoTime(next.to_time);
    if (next.from_time && fromSeconds !== null) query.set("from_time", String(fromSeconds));
    if (next.to_time && toSeconds !== null) query.set("to_time", String(toSeconds));
    try {
      setRows(await fetchTracks(query));
      setCurrentPage(1);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to load tracks.");
    }
  };

  useEffect(() => {
    const initialFilters = { ...DEFAULT_FILTERS, run_id: initialRunId };
    void fetchFilterOptions(initialFilters.run_id).then(setOptions);
    void load(initialFilters);
  }, []);

  useEffect(() => {
    void fetchFilterOptions(filters.run_id).then(setOptions);
  }, [filters.run_id]);

  useEffect(() => {
    if (trackingView !== "reconciled") return;
    let active = true;
    setReconciliationError(null);
    void fetchTrackReconciliation(filters.run_id || "latest")
      .then((payload) => {
        if (!active) return;
        setReconciliation(payload);
        setCurrentPage(1);
      })
      .catch((loadError) => {
        if (!active) return;
        setReconciliation(null);
        setReconciliationError(loadError instanceof Error ? loadError.message : "Failed to load reconciliation output.");
      });
    return () => {
      active = false;
    };
  }, [filters.run_id, trackingView]);

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    void load();
  };

  const onNaturalLanguageSearch = async (event: FormEvent) => {
    event.preventDefault();
    const query = nlQuery.trim();
    if (!query || isSearching) return;
    setIsSearching(true);
    setNlError(null);
    setNlResult(null);
    try {
      const result = await searchVehicles({ query, run_id: filters.run_id || "latest" });
      setNlResult(result);
    } catch (searchError) {
      setNlError(describeVehicleSearchError(searchError));
    } finally {
      setIsSearching(false);
    }
  };

  const onClear = () => {
    setFilters(DEFAULT_FILTERS);
    void fetchFilterOptions(DEFAULT_FILTERS.run_id).then(setOptions);
    void load(DEFAULT_FILTERS);
  };

  const totalPages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const pagedRows = useMemo(() => rows.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE), [currentPage, rows]);
  const reconciledRows = useMemo(() => buildReconciledRows(reconciliation), [reconciliation]);
  const reconciledTotalPages = Math.max(1, Math.ceil(reconciledRows.length / PAGE_SIZE));
  const pagedReconciledRows = useMemo(() => reconciledRows.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE), [currentPage, reconciledRows]);
  const activeChips = getActiveChips(filters);

  return (
    <section className="page-stack">
      <section className="card">
        <div className="section-heading">
          <div>
            <h2>Vehicles</h2>
            <p className="muted">Filter real tracks from saved runs or the current runtime.</p>
          </div>
        </div>

        <form className="nl-search-form" onSubmit={onNaturalLanguageSearch}>
          <label>
            <span>Natural-language search</span>
            <input
              value={nlQuery}
              onChange={(event) => setNlQuery(event.target.value)}
              placeholder='Search vehicles, e.g. "Show white cars between 5 and 10 seconds"'
              aria-label="Natural-language vehicle search"
            />
          </label>
          <button type="submit" disabled={isSearching || !nlQuery.trim()} aria-label="Run natural-language vehicle search">
            {isSearching ? "Searching..." : "Search"}
          </button>
        </form>

        {nlError ? <div className="empty-state search-error">{nlError}</div> : null}
        {nlResult ? <VehicleSearchResult result={nlResult} /> : null}

        <form className="filter-grid enhanced-filters" onSubmit={onSubmit}>
          <label>
            <span>Run</span>
            <select value={filters.run_id} onChange={(event) => setFilters({ ...filters, run_id: event.target.value })}>
              {(options.runs.length ? options.runs : ["latest"]).map((run) => (
                <option key={run} value={run}>{run === "latest" ? "Latest" : run}</option>
              ))}
            </select>
          </label>
          <label>
            <span>Camera</span>
            <select value={filters.camera_id} onChange={(event) => setFilters({ ...filters, camera_id: event.target.value })}>
              <option value="">All Cameras</option>
              {options.cameras.map((camera) => <option key={camera} value={camera}>{camera}</option>)}
            </select>
          </label>
          <label>
            <span>Class</span>
            <select value={filters.vehicle_class} onChange={(event) => setFilters({ ...filters, vehicle_class: event.target.value })}>
              <option value="">All Classes</option>
              {options.vehicle_classes.map((vehicleClass) => <option key={vehicleClass} value={vehicleClass}>{vehicleClass}</option>)}
            </select>
          </label>
          <label>
            <span>Colour</span>
            <select value={filters.colour} onChange={(event) => setFilters({ ...filters, colour: event.target.value })}>
              <option value="">All Colours</option>
              {options.colours.map((colour) => <option key={colour} value={colour}>{colour}</option>)}
            </select>
          </label>
          <label>
            <span>From</span>
            <input placeholder="00:00" value={filters.from_time} onChange={(event) => setFilters({ ...filters, from_time: event.target.value })} />
          </label>
          <label>
            <span>To</span>
            <input placeholder="01:00" value={filters.to_time} onChange={(event) => setFilters({ ...filters, to_time: event.target.value })} />
          </label>
          <label className="filter-track-id">
            <span>Track ID</span>
            <input placeholder="TRACK_5" value={filters.track_id} onChange={(event) => setFilters({ ...filters, track_id: event.target.value })} />
          </label>
          <div className="filter-actions">
            <button type="submit">Search</button>
            <button type="button" className="secondary-button" onClick={onClear}>Clear Filters</button>
          </div>
        </form>

        {activeChips.length > 0 ? (
          <div className="chip-row">
            {activeChips.map((chip) => <span key={chip} className="filter-chip">{chip}</span>)}
          </div>
        ) : null}

        <div className="table-toolbar">
          <div>
            <strong>Tracking View</strong>
            <span className="muted"> Raw tracks are production data. Reconciled vehicles are experiment-only.</span>
          </div>
          <div className="grid-controls" role="group" aria-label="Tracking View">
            <button type="button" className={`chip-button ${trackingView === "raw" ? "active" : ""}`} onClick={() => { setTrackingView("raw"); setCurrentPage(1); }}>
              Raw Tracks
            </button>
            <button type="button" className={`chip-button ${trackingView === "reconciled" ? "active" : ""}`} onClick={() => { setTrackingView("reconciled"); setCurrentPage(1); }}>
              Reconciled Vehicles
            </button>
          </div>
        </div>

        <div hidden={trackingView !== "raw"}>
        <div className="table-toolbar">
          <strong>{rows.length} vehicles found</strong>
          <span className="muted">Run context: {filters.run_id === "latest" ? "Latest" : filters.run_id || "All Runs"}</span>
        </div>
        {error ? <div className="empty-state">{error}</div> : null}

        {rows.length === 0 ? (
          <div className="empty-state">No tracks found for the selected filters.</div>
        ) : (
          <>
            <div className="table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Evidence</th>
                    <th>Run</th>
                    <th>Camera</th>
                    <th>Track</th>
                    <th>Class</th>
                    <th>Colour</th>
                    <th>First Seen</th>
                    <th>Last Seen</th>
                    <th>Duration</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {pagedRows.map((row) => (
                    <tr key={`${row.run_id ?? "runtime"}-${row.local_track_id}`}>
                      <td>{row.best_crop_url ? <img src={row.best_crop_url} alt={`${row.local_track_id} crop`} className="table-thumb" /> : <div className="thumb-placeholder small">No crop</div>}</td>
                      <td>{row.run_id ?? "runtime"}</td>
                      <td>{row.camera_id}</td>
                      <td><Link to={`/tracks/${row.camera_id}/${row.track_id}?run_id=${encodeURIComponent(row.run_id ?? "latest")}`}>{row.track_id}</Link></td>
                      <td>{(row.vehicle_class ?? "UNKNOWN").toUpperCase()}</td>
                      <td><span className="colour-badge">{row.colour ?? row.colour_status ?? "Unavailable"}</span></td>
                      <td>{formatVideoTime(row.first_seen_seconds ?? row.first_seen)}</td>
                      <td>{formatVideoTime(row.last_seen_seconds ?? row.last_seen)}</td>
                      <td>{formatVideoTime(row.duration_seconds)}</td>
                      <td>{row.status ?? "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="pagination-bar">
              <span>Showing {(currentPage - 1) * PAGE_SIZE + 1}–{Math.min(currentPage * PAGE_SIZE, rows.length)} of {rows.length}</span>
              <div className="pagination-actions">
                <button type="button" className="secondary-button" disabled={currentPage <= 1} onClick={() => setCurrentPage((page) => Math.max(1, page - 1))}>Previous</button>
                <span>Page {currentPage} / {totalPages}</span>
                <button type="button" className="secondary-button" disabled={currentPage >= totalPages} onClick={() => setCurrentPage((page) => Math.min(totalPages, page + 1))}>Next</button>
              </div>
            </div>
          </>
        )}
        </div>

        {trackingView === "reconciled" ? (
          <ReconciledVehiclesView
            reconciliation={reconciliation}
            error={reconciliationError}
            rows={pagedReconciledRows}
            allRowsCount={reconciledRows.length}
            currentPage={currentPage}
            totalPages={reconciledTotalPages}
            setCurrentPage={setCurrentPage}
          />
        ) : null}
      </section>
    </section>
  );
}

function ReconciledVehiclesView({
  reconciliation,
  error,
  rows,
  allRowsCount,
  currentPage,
  totalPages,
  setCurrentPage,
}: {
  reconciliation: TrackReconciliationResult | null;
  error: string | null;
  rows: ReconciledVehicleRow[];
  allRowsCount: number;
  currentPage: number;
  totalPages: number;
  setCurrentPage: (updater: (page: number) => number) => void;
}) {
  if (error) {
    return <div className="empty-state">{error}</div>;
  }
  if (reconciliation === null) {
    return <div className="empty-state">Loading reconciliation output...</div>;
  }
  if (!reconciliation.available) {
    return (
      <div className="empty-state">
        <p>{reconciliation.message || "Reconciliation test not available for this run."}</p>
        <code>{"python scripts/run_track_reconciliation_test.py --run-dir outputs\\runs\\<RUN_ID> --config config\\track_reconciliation_test.yaml"}</code>
      </div>
    );
  }
  const metrics = reconciliation.metrics ?? {};
  return (
    <>
      <div className="summary-grid reconciliation-summary">
        <MetricCard label="Raw ByteTrack Tracks" value={metrics.raw_bytetrack_unique_tracks} />
        <MetricCard label="Reconciled Vehicles" value={metrics.reconciled_vehicle_identities} />
        <MetricCard label="Recovered Fragments" value={metrics.track_fragments_merged ?? metrics.potential_duplicate_tracks_removed} />
        <MetricCard label="Accepted Matches" value={metrics.accepted_matches} />
        <MetricCard label="Ambiguous Matches" value={metrics.ambiguous_matches} />
      </div>

      {rows.length === 0 ? (
        <div className="empty-state">No reconciled vehicles found in this experiment output.</div>
      ) : (
        <>
          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Vehicle ID</th>
                  <th>Track Fragments</th>
                  <th>Class</th>
                  <th>Colour</th>
                  <th>Occlusion Recovery</th>
                  <th>Gap</th>
                  <th>Score</th>
                  <th>Second Best</th>
                  <th>Status</th>
                  <th>Evidence</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.vehicle_id}>
                    <td><span className="status">{row.vehicle_id}</span></td>
                    <td>
                      <div className="search-id-list">
                        {row.fragments.map((fragment) => <code key={fragment.local_track_id}>{shortTrackId(fragment.local_track_id)}</code>)}
                      </div>
                      <span className="muted">{row.fragments.map((fragment) => shortTrackId(fragment.local_track_id)).join(" -> ")}</span>
                    </td>
                    <td>{row.vehicle_class}</td>
                    <td><span className="colour-badge">{row.colour}</span></td>
                    <td>{row.recovered ? "YES" : "NO"}</td>
                    <td>{row.recovered ? `${formatNumber(row.gap_frames)} frames / ${formatSeconds(row.gap_seconds)}` : "-"}</td>
                    <td>{row.recovered ? formatScore(row.score) : "-"}</td>
                    <td>{row.recovered ? formatScore(row.second_best_score) : "-"}</td>
                    <td>{row.result}</td>
                    <td>{row.recovered ? <Link to={`/runs/${encodeURIComponent(reconciliation.run_id)}/reconciliation`}>View Evidence</Link> : "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Pagination allRowsCount={allRowsCount} currentPage={currentPage} totalPages={totalPages} setCurrentPage={setCurrentPage} />
        </>
      )}

      <section className="card">
        <div className="section-heading">
          <div>
            <h3>Accepted Associations</h3>
            <p className="muted">Read-only test associations from the reconciliation output.</p>
          </div>
        </div>
        {reconciliation.accepted_associations.length === 0 ? (
          <div className="empty-state">No accepted associations.</div>
        ) : (
          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Old Track</th>
                  <th>New Track</th>
                  <th>Vehicle ID</th>
                  <th>Class</th>
                  <th>Colour</th>
                  <th>Gap</th>
                  <th>Score</th>
                  <th>Second Best</th>
                  <th>Result</th>
                </tr>
              </thead>
              <tbody>
                {reconciliation.accepted_associations.map((item) => (
                  <tr key={`${item.old_track}->${item.new_track}`}>
                    <td>{shortTrackId(item.old_track)}</td>
                    <td>{shortTrackId(item.new_track)}</td>
                    <td>{item.vehicle_id}</td>
                    <td>{item.class ?? "UNKNOWN"}</td>
                    <td>{item.colour ?? "UNKNOWN"}</td>
                    <td>{formatSeconds(item.gap_seconds)}</td>
                    <td>{formatScore(item.score)}</td>
                    <td>{formatScore(item.second_best_score)}</td>
                    <td>{item.result ?? "RECOVERED"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}

function Pagination({
  allRowsCount,
  currentPage,
  totalPages,
  setCurrentPage,
}: {
  allRowsCount: number;
  currentPage: number;
  totalPages: number;
  setCurrentPage: (updater: (page: number) => number) => void;
}) {
  return (
    <div className="pagination-bar">
      <span>Showing {(currentPage - 1) * PAGE_SIZE + 1}-{Math.min(currentPage * PAGE_SIZE, allRowsCount)} of {allRowsCount}</span>
      <div className="pagination-actions">
        <button type="button" className="secondary-button" disabled={currentPage <= 1} onClick={() => setCurrentPage((page) => Math.max(1, page - 1))}>Previous</button>
        <span>Page {currentPage} / {totalPages}</span>
        <button type="button" className="secondary-button" disabled={currentPage >= totalPages} onClick={() => setCurrentPage((page) => Math.min(totalPages, page + 1))}>Next</button>
      </div>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="card summary-card">
      <span className="summary-label">{label}</span>
      <strong className="summary-value">{formatNumber(value)}</strong>
    </div>
  );
}

function VehicleSearchResult({ result }: { result: VehicleSearchResponse }) {
  const total = typeof result.analytics_result.total === "number" ? result.analytics_result.total : null;
  const vehicleIds = result.analytics_result.vehicle_ids ?? [];
  const classesPresent = result.analytics_result.vehicle_classes_present ?? [];
  const coloursPresent = result.analytics_result.colours_present ?? [];
  return (
    <section className="nl-search-result" aria-label="Vehicle search result">
      <p className="search-answer">{result.response}</p>
      {total !== null ? <strong>Matches: {total}</strong> : null}
      {vehicleIds.length > 0 ? (
        <div className="search-id-list">
          {vehicleIds.map((vehicleId) => (
            <code key={vehicleId}>{vehicleId}</code>
          ))}
        </div>
      ) : null}
      {classesPresent.length > 0 ? <p className="muted">Classes: {classesPresent.join(", ")}</p> : null}
      {coloursPresent.length > 0 ? <p className="muted">Colours: {coloursPresent.join(", ")}</p> : null}
      <details className="parsed-query-details">
        <summary>Parsed query</summary>
        <dl>
          <div><dt>Intent</dt><dd>{result.parsed_query.intent}</dd></div>
          <div><dt>Class</dt><dd>{result.parsed_query.vehicle_class ?? "Any"}</dd></div>
          <div><dt>Colour</dt><dd>{result.parsed_query.colour ?? "Any"}</dd></div>
          <div><dt>Start</dt><dd>{formatOptionalNumber(result.parsed_query.start_time)}</dd></div>
          <div><dt>End</dt><dd>{formatOptionalNumber(result.parsed_query.end_time)}</dd></div>
          <div><dt>Camera</dt><dd>{result.parsed_query.camera_id ?? "Any"}</dd></div>
          <div><dt>Run</dt><dd>{result.run_id}</dd></div>
        </dl>
      </details>
    </section>
  );
}

function formatOptionalNumber(value: number | null): string {
  return typeof value === "number" ? value.toFixed(1) : "Any";
}

function describeVehicleSearchError(error: unknown): string {
  if (error instanceof ApiError) {
    const detail = error.detail as { detail?: unknown } | null;
    const payload = typeof detail?.detail === "object" && detail.detail !== null ? detail.detail as { error?: string; detail?: string } : null;
    if (error.status === 400 && payload?.error === "query_not_understood") {
      return payload.detail || "I couldn't understand that query. Try specifying a vehicle type, colour, or time range.";
    }
    if (error.status === 404) {
      return "Run not found.";
    }
    if (error.status >= 500) {
      return "Vehicle search is temporarily unavailable.";
    }
  }
  return error instanceof Error ? error.message : "Vehicle search failed.";
}
