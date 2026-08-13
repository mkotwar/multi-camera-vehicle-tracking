import { Link } from "react-router-dom";
import { useEffect, useState } from "react";
import { fetchRuns } from "../api/runs";
import type { RunSummary } from "../types/run";
import { formatRelativeDate, formatVideoTime } from "../utils/time";

export function RunsPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);

  useEffect(() => {
    void fetchRuns().then(setRuns);
  }, []);

  return (
    <section className="card">
      <div className="section-heading">
        <div>
          <h2>Runs</h2>
          <p className="muted">Available pipeline runs from `outputs/runs`.</p>
        </div>
      </div>

      {runs.length === 0 ? (
        <div className="empty-state">No runs found.</div>
      ) : (
        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Run ID</th>
                <th>Status</th>
                <th>Cameras</th>
                <th>Tracks</th>
                <th>Total Frames</th>
                <th>Started</th>
                <th>Duration</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.run_id}>
                  <td>{run.run_id}</td>
                  <td><span className="status">{run.status}</span></td>
                  <td>{run.camera_count ?? "Unavailable"}</td>
                  <td>{run.track_count ?? "Unavailable"}</td>
                  <td>{run.processed_frames ?? "Unavailable"}</td>
                  <td>{formatRelativeDate(run.start_time)}</td>
                  <td>{formatVideoTime(run.duration_seconds)}</td>
                  <td>
                    <div className="run-actions">
                      <Link to={`/vehicles?run_id=${encodeURIComponent(run.run_id)}`}>Vehicles</Link>
                      <Link to={`/runs/${encodeURIComponent(run.run_id)}/reconciliation`}>Reconciliation</Link>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
