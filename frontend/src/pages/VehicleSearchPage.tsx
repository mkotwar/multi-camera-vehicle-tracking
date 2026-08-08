import { FormEvent, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useSearchParams } from "react-router-dom";
import { fetchFilterOptions } from "../api/filters";
import { fetchTracks } from "../api/tracks";
import type { FilterOptions } from "../types/filters";
import type { TrackRecord } from "../types/track";
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

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    void load();
  };

  const onClear = () => {
    setFilters(DEFAULT_FILTERS);
    void fetchFilterOptions(DEFAULT_FILTERS.run_id).then(setOptions);
    void load(DEFAULT_FILTERS);
  };

  const totalPages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const pagedRows = useMemo(() => rows.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE), [currentPage, rows]);
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
      </section>
    </section>
  );
}
