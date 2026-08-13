import { useParams, useSearchParams } from "react-router-dom";
import { TrackDetailPanel } from "../components/track/TrackDetailPanel";

export function TrackDetailPage() {
  const { cameraId = "", trackId = "" } = useParams();
  const [searchParams] = useSearchParams();
  const runId = searchParams.get("run_id");

  return (
    <section className="page-stack">
      <TrackDetailPanel selection={{ cameraId, trackId, runId }} />
    </section>
  );
}
