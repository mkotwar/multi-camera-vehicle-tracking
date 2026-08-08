import type { SystemStatus } from "../../types/system";

type Props = { system: SystemStatus | null };

export function SystemPanel({ system }: Props) {
  if (!system) {
    return <section className="card">Loading system status…</section>;
  }

  return (
    <section className="card system-panel">
      <div className="section-heading">
        <div>
          <h3>Technical Health</h3>
          <p className="muted">Developer-facing metrics from the current runtime or latest saved run.</p>
        </div>
      </div>
      <div className="stats-grid compact-stats">
        <div><strong>Pipeline</strong><span>{system.pipeline_status || "Unavailable"}</span></div>
        <div><strong>Processed FPS</strong><span>{system.processed_fps ? system.processed_fps.toFixed(2) : "Unavailable"}</span></div>
        <div><strong>YOLO</strong><span>{system.yolo_status || "Unavailable"}</span></div>
        <div><strong>YOLO ImgSz</strong><span>{system.yolo_image_size ?? "Unavailable"}</span></div>
        <div><strong>Batch Size</strong><span>{system.yolo_batch_size ?? "Unavailable"}</span></div>
        <div><strong>Colour Worker</strong><span>{system.colour_worker_status || "Unavailable"}</span></div>
        <div><strong>Queue Depth</strong><span>{system.colour_queue_depth}/{system.colour_queue_capacity || "Unavailable"}</span></div>
        <div><strong>Pending Jobs</strong><span>{system.pending_colour_jobs}</span></div>
        <div><strong>Peak Queue</strong><span>{system.colour_queue_peak_depth ?? "Unavailable"}</span></div>
        <div><strong>Avg Calls/Track</strong><span>{system.average_colour_calls_per_track ?? "Unavailable"}</span></div>
        <div><strong>Cache Misses</strong><span>{system.cache_misses}</span></div>
        <div><strong>Frame Loss</strong><span>{system.frame_loss}</span></div>
        <div><strong>Order Violations</strong><span>{system.order_violations}</span></div>
      </div>
    </section>
  );
}
