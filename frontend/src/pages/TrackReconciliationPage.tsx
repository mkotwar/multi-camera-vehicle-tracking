import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { fetchTrackReconciliation } from "../api/runs";
import { apiUrl } from "../api/client";
import type { TrackReconciliationResult } from "../types/run";

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

function shortTrackId(value: string): string {
  const parts = value.split(":");
  return parts[parts.length - 1] || value;
}

export function TrackReconciliationPage() {
  const { runId = "latest" } = useParams();
  const [data, setData] = useState<TrackReconciliationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    void fetchTrackReconciliation(runId)
      .then((payload) => {
        if (active) {
          setData(payload);
        }
      })
      .catch(() => {
        if (active) {
          setError("Track reconciliation output is not available for this run.");
          setData(null);
        }
      })
      .finally(() => {
        if (active) {
          setLoading(false);
        }
      });
    return () => {
      active = false;
    };
  }, [runId]);

  const manualByPair = useMemo(() => {
    const rows = new Map<string, string>();
    for (const item of data?.manual_validation ?? []) {
      rows.set(`${item.old_track}->${item.new_track}`, item.manual_label || "UNCERTAIN");
    }
    return rows;
  }, [data]);

  const visualByPair = useMemo(() => {
    const rows = new Map<string, NonNullable<TrackReconciliationResult["visual_evidence"]>[number]>();
    for (const item of data?.visual_evidence ?? []) {
      rows.set(item.pair_key, item);
    }
    return rows;
  }, [data]);

  if (loading) {
    return <section className="card track-detail-state">Loading reconciliation output...</section>;
  }

  if (error || data === null || !data.available) {
    return (
      <section className="card track-detail-state error">
        <h2>Track Reconciliation</h2>
        <p>{error || data?.message || "Reconciliation test has not been run for this run."}</p>
        <code>{"python scripts/run_track_reconciliation_test.py --run-dir outputs\\runs\\<RUN_ID> --config config\\track_reconciliation_test.yaml"}</code>
        <Link to="/runs">Back to runs</Link>
      </section>
    );
  }

  const metrics = data.metrics ?? {};

  return (
    <section className="page-stack reconciliation-page">
      <div className="section-heading">
        <div>
          <h2>Track Reconciliation</h2>
          <p className="muted">Run {data.run_id}. This is experiment-only output; production track IDs remain unchanged.</p>
        </div>
        <Link className="secondary-button" to={`/vehicles?run_id=${encodeURIComponent(data.run_id)}`}>Open vehicles</Link>
      </div>

      <div className="summary-grid reconciliation-summary">
        <div className="card summary-card">
          <span className="summary-label">Raw Tracks</span>
          <strong className="summary-value">{formatNumber(metrics.raw_bytetrack_unique_tracks)}</strong>
        </div>
        <div className="card summary-card">
          <span className="summary-label">Vehicle IDs</span>
          <strong className="summary-value">{formatNumber(metrics.reconciled_vehicle_identities)}</strong>
        </div>
        <div className="card summary-card">
          <span className="summary-label">Potential Duplicates</span>
          <strong className="summary-value">{formatNumber(metrics.potential_duplicate_tracks_removed)}</strong>
        </div>
        <div className="card summary-card">
          <span className="summary-label">Attempts</span>
          <strong className="summary-value">{formatNumber(metrics.reconciliation_attempts)}</strong>
        </div>
        <div className="card summary-card">
          <span className="summary-label">Accepted</span>
          <strong className="summary-value">{formatNumber(metrics.accepted_matches)}</strong>
        </div>
        <div className="card summary-card">
          <span className="summary-label">Ambiguous</span>
          <strong className="summary-value">{formatNumber(metrics.ambiguous_matches)}</strong>
        </div>
      </div>

      <section className="card">
        <div className="section-heading">
          <div>
            <h3>Accepted Associations</h3>
            <p className="muted">Review these manually before trusting any reduction in unique vehicles.</p>
          </div>
        </div>
        {data.accepted_associations.length === 0 ? (
          <div className="empty-state">No accepted recoveries in this experiment output.</div>
        ) : (
          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Old Track</th>
                  <th>New Track</th>
                  <th>Vehicle ID</th>
                  <th>Gap</th>
                  <th>Score</th>
                  <th>Second Best</th>
                  <th>Class</th>
                  <th>Colour</th>
                  <th>Manual</th>
                </tr>
              </thead>
              <tbody>
                {data.accepted_associations.map((item) => {
                  const pairKey = `${item.old_track}->${item.new_track}`;
                  return (
                    <tr key={pairKey}>
                      <td>{shortTrackId(item.old_track)}</td>
                      <td>{shortTrackId(item.new_track)}</td>
                      <td><span className="status">{item.vehicle_id}</span></td>
                      <td>{formatNumber(item.gap_frames)} frames</td>
                      <td>{formatScore(item.score)}</td>
                      <td>{formatScore(item.second_best_score)}</td>
                      <td>{item.class ?? "UNKNOWN"}</td>
                      <td><span className="colour-badge">{item.colour ?? "UNKNOWN"}</span></td>
                      <td>{manualByPair.get(pairKey) ?? "UNCERTAIN"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="card">
        <div className="section-heading">
          <div>
            <h3>Visual Evidence</h3>
            <p className="muted">Contact sheets show the last frame before loss and the first frame after recovery.</p>
          </div>
        </div>
        <div className="reconciliation-evidence-grid">
          {data.accepted_associations.map((item) => {
            const pairKey = `${item.old_track.replace(/:/g, "_")}__${item.new_track.replace(/:/g, "_")}`;
            const visual = visualByPair.get(pairKey);
            return (
              <article className="reconciliation-evidence-card" key={pairKey}>
                <div className="evidence-row-heading">
                  <div>
                    <strong>{shortTrackId(item.old_track)}{" -> "}{shortTrackId(item.new_track)}</strong>
                    <p className="muted">{item.vehicle_id} | score {formatScore(item.score)}</p>
                  </div>
                  <span className="status">{item.result ?? "ACCEPTED"}</span>
                </div>
                {visual?.contact_sheet_url ? (
                  <a href={apiUrl(visual.contact_sheet_url)} target="_blank" rel="noreferrer">
                    <img className="reconciliation-contact-sheet" src={apiUrl(visual.contact_sheet_url)} alt={`${item.old_track} to ${item.new_track}`} />
                  </a>
                ) : (
                  <div className="empty-inline">No contact sheet was saved for this pair.</div>
                )}
              </article>
            );
          })}
        </div>
      </section>
    </section>
  );
}
