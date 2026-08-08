import { Link } from "react-router-dom";
import { apiUrl } from "../../api/client";
import type { CameraState } from "../../types/camera";
import { formatRelativeDate, formatVideoTime } from "../../utils/time";

type Props = { camera: CameraState };

export function CameraCard({ camera }: Props) {
  const frameUrl = camera.frame_url
    ? apiUrl(camera.frame_url)
    : apiUrl(`/api/cameras/${camera.camera_id}/frame?ts=${Date.now()}${camera.run_id ? `&run_id=${encodeURIComponent(camera.run_id)}` : ""}`);

  return (
    <section className="card camera-card">
      <div className="card-header">
        <div>
          <h3>{camera.camera_id}</h3>
          <p className="muted">{camera.source_type === "saved_run" ? "Saved run view" : "Live processed view"}</p>
        </div>
        <span className={`status status-${camera.status}`}>{camera.status}</span>
      </div>
      <img className="camera-frame" src={frameUrl} alt={`${camera.camera_id} frame`} />
      <div className="camera-meta">
        <span>Frame: {camera.frame_number ?? "Unavailable"}</span>
        <span>FPS: {camera.processed_fps ? camera.processed_fps.toFixed(2) : "Unavailable"}</span>
        <span>Active: {camera.active_vehicle_count}</span>
        <span>{camera.timestamp_seconds != null ? formatVideoTime(camera.timestamp_seconds) : formatRelativeDate(camera.last_update)}</span>
      </div>
      <div className="detection-list">
        {camera.detections.slice(0, 5).map((detection) => (
          <Link
            key={`${camera.camera_id}-${detection.track_id}`}
            to={`/tracks/${camera.camera_id}/${detection.track_id}`}
            className="detection-pill"
          >
            <strong>{(detection.vehicle_class ?? "UNKNOWN").toUpperCase()}</strong> · {detection.track_id} · {detection.colour ?? "Colour pending"}
          </Link>
        ))}
        {camera.detections.length === 0 ? <div className="empty-inline">No active detections in this camera view.</div> : null}
      </div>
    </section>
  );
}
