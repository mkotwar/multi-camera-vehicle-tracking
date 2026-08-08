import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { fetchCameras } from "../api/cameras";
import { fetchRuns } from "../api/runs";
import { fetchSystemStatus } from "../api/system";
import { fetchTracks } from "../api/tracks";
import { CameraCard } from "../components/camera/CameraCard";
import { SystemPanel } from "../components/common/SystemPanel";
import { useLiveUpdates } from "../hooks/useLiveUpdates";
import type { CameraState } from "../types/camera";
import type { RunSummary } from "../types/run";
import type { SystemStatus } from "../types/system";
import type { TrackRecord } from "../types/track";
import { formatVideoTime } from "../utils/time";

const GRID_OPTIONS = [
  { label: "Auto", value: "auto" },
  { label: "2 columns", value: "2" },
  { label: "3 columns", value: "3" },
  { label: "4 columns", value: "4" },
] as const;

function summaryValue(value: string | number | null | undefined): string {
  if (value === null || value === undefined || value === "") {
    return "Unavailable";
  }
  return String(value);
}

export function DashboardPage() {
  const [cameras, setCameras] = useState<CameraState[]>([]);
  const [system, setSystem] = useState<SystemStatus | null>(null);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [recentTracks, setRecentTracks] = useState<TrackRecord[]>([]);
  const [cameraColumns, setCameraColumns] = useState<(typeof GRID_OPTIONS)[number]["value"]>("auto");
  const [showTechnicalHealth, setShowTechnicalHealth] = useState(false);

  useEffect(() => {
    void fetchCameras("latest").then(setCameras);
    void fetchSystemStatus().then(setSystem);
    void fetchRuns().then(setRuns);
    const query = new URLSearchParams({ run_id: "latest" });
    void fetchTracks(query).then((rows) => setRecentTracks(rows.slice(0, 12)));
  }, []);

  useLiveUpdates({
    onSnapshot: (payload) => {
      if (Array.isArray(payload.cameras)) setCameras(payload.cameras as CameraState[]);
      if (payload.system) setSystem(payload.system as SystemStatus);
    },
    onCameraUpdate: (payload) => {
      setCameras((current) =>
        current.map((camera) =>
          camera.camera_id === payload.camera_id
            ? {
                ...camera,
                ...payload,
                frame_number: Number(payload.frame_number),
                processed_fps: Number(payload.processed_fps),
                active_vehicle_count: Number(payload.active_vehicle_count),
              }
            : camera,
        ),
      );
    },
    onSystemStatus: (payload) => setSystem(payload as unknown as SystemStatus),
    onTrackColourUpdate: () => {
      const query = new URLSearchParams({ run_id: "latest" });
      void fetchTracks(query).then((rows) => setRecentTracks(rows.slice(0, 12)));
    },
  });

  const latestRun = runs[0] ?? null;
  const currentTrackCount = system?.track_count ?? (recentTracks.length || latestRun?.track_count);
  const summaryCards = useMemo(
    () => [
      { label: "Pipeline Status", value: summaryValue(system?.pipeline_status?.toUpperCase()) },
      { label: "Cameras Active / Total", value: `${system?.processing_camera_count ?? cameras.length} / ${system?.camera_count ?? cameras.length}` },
      { label: "Vehicles / Tracks", value: summaryValue(currentTrackCount) },
      { label: "Processed FPS", value: system?.processed_fps ? system.processed_fps.toFixed(2) : "Unavailable" },
      { label: "YOLO", value: summaryValue(system?.yolo_status?.toUpperCase()) },
      { label: "Colour Worker", value: summaryValue(system?.colour_worker_status?.toUpperCase()) },
    ],
    [cameras.length, currentTrackCount, system],
  );

  return (
    <div className="dashboard-page">
      <section className="summary-grid">
        {summaryCards.map((card) => (
          <article key={card.label} className="card summary-card">
            <span className="summary-label">{card.label}</span>
            <strong className="summary-value">{card.value}</strong>
          </article>
        ))}
      </section>

      <section className="card">
        <div className="section-heading">
          <div>
            <h3>Camera Wall</h3>
            <p className="muted">Latest real processed frames from the current runtime or latest saved run.</p>
          </div>
          <div className="grid-controls">
            {GRID_OPTIONS.map((option) => (
              <button
                key={option.value}
                type="button"
                className={`chip-button ${cameraColumns === option.value ? "active" : ""}`}
                onClick={() => setCameraColumns(option.value)}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>
        {cameras.length === 0 ? (
          <div className="empty-state">No cameras available from the active runtime or latest saved run.</div>
        ) : (
          <section className={`camera-grid columns-${cameraColumns}`}>
            {cameras.map((camera) => (
              <CameraCard key={camera.camera_id} camera={camera} />
            ))}
          </section>
        )}
      </section>

      <section className="split-grid">
        <section className="card">
          <div className="section-heading">
            <div>
              <h3>Recent Vehicles</h3>
              <p className="muted">{latestRun ? `Showing latest tracks from run ${latestRun.run_id}` : "Showing latest available tracks."}</p>
            </div>
            <Link to="/vehicles" className="text-link">Open vehicles</Link>
          </div>
          <div className="recent-vehicles">
            {recentTracks.length === 0 ? (
              <div className="empty-state">No tracks found in the selected run view.</div>
            ) : (
              recentTracks.map((track) => (
                <Link
                  key={`${track.run_id ?? "runtime"}-${track.local_track_id}`}
                  to={`/tracks/${track.camera_id}/${track.track_id}?run_id=${encodeURIComponent(track.run_id ?? "latest")}`}
                  className="recent-vehicle-row"
                >
                  {track.best_crop_url ? <img src={track.best_crop_url} alt={`${track.local_track_id} thumbnail`} className="thumb-image" /> : <div className="thumb-placeholder">No crop</div>}
                  <div className="recent-vehicle-copy">
                    <strong>{(track.vehicle_class ?? "UNKNOWN").toUpperCase()} · {track.colour ?? track.colour_status ?? "Colour pending"}</strong>
                    <span>{track.camera_id} · {track.track_id}</span>
                  </div>
                  <span className="muted">{formatVideoTime(track.last_seen_seconds ?? track.last_seen)}</span>
                </Link>
              ))
            )}
          </div>
        </section>

        <section className="card">
          <div className="section-heading">
            <div>
              <h3>Recent Runs</h3>
              <p className="muted">Real run folders from `outputs/runs`.</p>
            </div>
            <Link to="/runs" className="text-link">Open runs</Link>
          </div>
          <div className="run-list">
            {runs.slice(0, 6).map((run) => (
              <div key={run.run_id} className="run-row">
                <div>
                  <strong>{run.run_id}</strong>
                  <div className="muted">{run.track_count ?? "Unavailable"} tracks · {run.camera_count ?? "Unavailable"} cameras</div>
                </div>
                <div className="align-right">
                  <span className="status">{run.status}</span>
                  <div className="muted">{run.processed_frames ?? "Unavailable"} frames</div>
                </div>
              </div>
            ))}
          </div>
        </section>
      </section>

      <section className="card">
        <div className="section-heading">
          <div>
            <h3>Technical Health</h3>
            <p className="muted">Available, but moved out of the main operator focus.</p>
          </div>
          <button type="button" className="chip-button" onClick={() => setShowTechnicalHealth((current) => !current)}>
            {showTechnicalHealth ? "Hide" : "Show"}
          </button>
        </div>
        {showTechnicalHealth ? <SystemPanel system={system} /> : <div className="empty-inline">Expand to view queue depth, cache misses, YOLO settings, and colour worker details.</div>}
      </section>
    </div>
  );
}
