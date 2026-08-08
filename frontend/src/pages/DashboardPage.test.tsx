import { render, screen, waitFor } from "@testing-library/react";
import { BrowserRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DashboardPage } from "./DashboardPage";

vi.mock("../api/cameras", () => ({
  fetchCameras: vi.fn().mockResolvedValue([{ camera_id: "CAM_001", name: "CAM_001", status: "completed", frame_number: 13, timestamp_seconds: 2.5, processed_fps: 0, input_fps: null, active_vehicle_count: 1, active_track_ids: ["TRACK_1"], detections: [], last_update: null, frame_url: "/api/media/tracked_frames/20260808_182124/CAM_001/frame_000013.jpg", run_id: "20260808_182124", source_type: "saved_run" }]),
}));

vi.mock("../api/runs", () => ({
  fetchRuns: vi.fn().mockResolvedValue([{ run_id: "20260808_182124", status: "COMPLETED", camera_count: 2, processed_frames: 40, track_count: 32 }]),
}));

vi.mock("../api/system", () => ({
  fetchSystemStatus: vi.fn().mockResolvedValue({ pipeline_status: "completed", camera_count: 2, processing_camera_count: 2, online_camera_count: 2, processed_fps: 4.2, yolo_status: "healthy", colour_worker_status: "healthy", colour_queue_depth: 0, colour_queue_capacity: 100, pending_colour_jobs: 0, cache_misses: 0, frame_loss: 0, order_violations: 0, last_update: "", track_count: 32 }),
}));

vi.mock("../api/tracks", () => ({
  fetchTracks: vi.fn().mockResolvedValue([{ run_id: "20260808_182124", camera_id: "CAM_001", track_id: "TRACK_1", local_track_id: "CAM_001:TRACK_1", vehicle_class: "CAR", colour: "WHITE", best_crop_url: "/crop.jpg", last_seen_seconds: 2.5 }]),
}));

vi.mock("../hooks/useLiveUpdates", () => ({
  useLiveUpdates: vi.fn(),
}));

describe("DashboardPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders summary cards and recent vehicles from real API payloads", async () => {
    render(
      <BrowserRouter>
        <DashboardPage />
      </BrowserRouter>,
    );

    await waitFor(() => expect(screen.getByText("Camera Wall")).toBeInTheDocument());
    expect(screen.getByText("Pipeline Status")).toBeInTheDocument();
    expect(screen.getByText("Recent Vehicles")).toBeInTheDocument();
    expect(screen.getByText("Vehicles / Tracks")).toBeInTheDocument();
    expect(screen.getByText(/TRACK_1/)).toBeInTheDocument();
  });
});
