import { useEffect, useMemo, useState } from "react";
import { useSearchParams, useParams } from "react-router-dom";
import { fetchTrack, fetchTrackEvidence } from "../api/tracks";
import type { EvidenceRecord, TrackRecord } from "../types/track";
import { formatVideoTime } from "../utils/time";

function labelForRole(role?: string | null): string {
  return String(role || "Evidence").split("_").join(" ");
}

export function TrackDetailPage() {
  const { cameraId = "", trackId = "" } = useParams();
  const [searchParams] = useSearchParams();
  const runId = searchParams.get("run_id") ?? undefined;
  const [track, setTrack] = useState<TrackRecord | null>(null);
  const [evidence, setEvidence] = useState<EvidenceRecord[]>([]);

  useEffect(() => {
    void fetchTrack(cameraId, trackId, runId).then(setTrack);
    void fetchTrackEvidence(cameraId, trackId, runId).then(setEvidence);
  }, [cameraId, runId, trackId]);

  const uniqueEvidence = useMemo(() => {
    const seen = new Set<string>();
    return evidence.filter((item) => {
      const key = `${item.frame_number}-${item.vehicle_crop_path}-${item.annotated_frame_path}`;
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  }, [evidence]);

  if (!track) {
    return <section className="card">Loading track…</section>;
  }

  const firstSeen = track.first_seen_seconds ?? track.first_seen;
  const lastSeen = track.last_seen_seconds ?? track.last_seen;
  const duration = track.duration_seconds;
  const timelineRange = Math.max(duration ?? 0, 0.001);

  return (
    <section className="page-stack">
      <section className="card track-detail">
        <div className="track-header">
          <div>
            <h2>{track.local_track_id}</h2>
            <p className="muted">Run {track.run_id ?? "runtime"} · Camera {track.camera_id}</p>
          </div>
          <span className="status">{track.status ?? "Unavailable"}</span>
        </div>

        <div className="stats-grid detail-overview">
          <div><strong>Vehicle Class</strong><span>{(track.vehicle_class ?? "UNKNOWN").toUpperCase()}</span></div>
          <div><strong>Colour</strong><span>{track.colour ?? track.colour_status ?? "Colour pending"}</span></div>
          <div><strong>First Seen</strong><span>{formatVideoTime(firstSeen)}</span></div>
          <div><strong>Last Seen</strong><span>{formatVideoTime(lastSeen)}</span></div>
          <div><strong>Duration</strong><span>{formatVideoTime(duration)}</span></div>
          <div><strong>Frame Range</strong><span>{track.first_frame ?? "Unavailable"} → {track.last_frame ?? "Unavailable"}</span></div>
        </div>

        <section className="timeline-card">
          <div className="timeline-labels">
            <span>{formatVideoTime(firstSeen)}</span>
            <span>{formatVideoTime(lastSeen)}</span>
          </div>
          <div className="timeline-bar">
            <span className="timeline-point start" />
            {uniqueEvidence.slice(0, 3).map((item, index) => {
              const seconds = Number(item.timestamp_seconds ?? firstSeen ?? 0);
              const offset = duration ? Math.max(0, Math.min(100, ((seconds - (firstSeen ?? 0)) / timelineRange) * 100)) : 50;
              return <span key={`${item.frame_number}-${index}`} className="timeline-point marker" style={{ left: `${offset}%` }} />;
            })}
            <span className="timeline-point end" />
          </div>
          <div className="timeline-labels muted">
            <span>First Seen</span>
            <span>Best Evidence / Key Frames</span>
            <span>Last Seen</span>
          </div>
        </section>
      </section>

      <section className="card">
        <div className="section-heading">
          <div>
            <h3>Evidence</h3>
            <p className="muted">Full annotated frame context alongside the exact crop used for the AI result.</p>
          </div>
        </div>
        {uniqueEvidence.length === 0 ? (
          <div className="empty-state">No evidence is available for this track.</div>
        ) : (
          <div className="evidence-stack">
            {uniqueEvidence.map((item, index) => (
              <article key={`${track.local_track_id}-${item.frame_number ?? index}-${item.evidence_role ?? "evidence"}`} className="evidence-row-card">
                <div className="evidence-row-heading">
                  <div>
                    <strong>{labelForRole(item.evidence_role)}</strong>
                    <div className="muted">Timestamp {formatVideoTime(item.timestamp_seconds)} · Frame {item.frame_number ?? "Unavailable"}</div>
                  </div>
                  <span className="colour-badge">{item.colour_crop_result ?? track.colour ?? "Unavailable"}</span>
                </div>
                <div className="evidence-pair-grid">
                  <div className="media-panel">
                    <span className="media-label">Full Frame</span>
                    {item.full_frame_url ? <img src={item.full_frame_url} alt={`${track.local_track_id} full frame`} className="detail-image full-frame" /> : <div className="detail-image placeholder">No full frame</div>}
                  </div>
                  <div className="media-panel">
                    <span className="media-label">Vehicle Crop</span>
                    {item.crop_url ? <img src={item.crop_url} alt={`${track.local_track_id} crop`} className="detail-image crop-frame" /> : <div className="detail-image placeholder">No crop</div>}
                  </div>
                </div>
              </article>
            ))}
          </div>
        )}
      </section>

      <details className="card technical-details">
        <summary>Technical Details</summary>
        <div className="stats-grid compact-stats">
          <div><strong>Observation Count</strong><span>{track.observation_count ?? "Unavailable"}</span></div>
          <div><strong>Completion Reason</strong><span>{track.completion_reason ?? "Unavailable"}</span></div>
          <div><strong>Colour Status</strong><span>{track.colour_status ?? "Unavailable"}</span></div>
          <div><strong>Best Crop</strong><span>{track.best_crop_url ? "Available" : "Unavailable"}</span></div>
        </div>
        {track.colour_resolution && track.colour_resolution.length > 0 ? (
          <div className="resolution-steps">
            {track.colour_resolution.map((step) => (
              <div key={`${step.index}-${step.frame_number ?? "na"}`} className="resolution-step">
                <strong>Crop {step.index}</strong>
                <span>{step.label ?? "UNKNOWN"} {step.index === 1 && track.colour ? "✓" : ""}</span>
                <span className="muted">Frame {step.frame_number ?? "Unavailable"} · {step.evidence_role ?? "Unavailable"}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-inline">No colour resolution detail is available in this run output.</div>
        )}
      </details>
    </section>
  );
}
