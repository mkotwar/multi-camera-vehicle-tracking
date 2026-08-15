import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { TrackDetailPage } from "./TrackDetailPage";

vi.mock("../api/tracks", () => ({
  fetchTrack: vi.fn().mockResolvedValue({
    run_id: "20260808_182124",
    camera_id: "CAM_002",
    track_id: "TRACK_5",
    local_track_id: "CAM_002:TRACK_5",
    vehicle_class: "3WHEELER",
    colour: "GREEN",
    first_seen_seconds: 2.1,
    last_seen_seconds: 7.45,
    duration_seconds: 5.35,
    first_frame: 2,
    last_frame: 19,
    status: "COMPLETED",
    plate_text: "HR38AD4296",
    plate_detected: true,
    plate_text_confidence: 0.82,
    colour_resolution: [{ index: 1, label: "GREEN", frame_number: 12, evidence_role: "BEST_OVERALL" }],
  }),
  fetchTrackEvidence: vi.fn().mockResolvedValue([
    {
      frame_number: 12,
      timestamp_seconds: 4.72,
      evidence_role: "BEST_OVERALL",
      crop_url: "/crop.jpg",
      full_frame_url: "/frame.jpg",
      colour_crop_result: "GREEN",
    },
  ]),
}));

describe("TrackDetailPage", () => {
  it("renders full frame, crop, and timing details", async () => {
    render(
      <MemoryRouter initialEntries={["/tracks/CAM_002/TRACK_5?run_id=20260808_182124"]}>
        <Routes>
          <Route path="/tracks/:cameraId/:trackId" element={<TrackDetailPage />} />
        </Routes>
      </MemoryRouter>,
    );

    await waitFor(() => expect(screen.getByText("CAM_002:TRACK_5")).toBeInTheDocument());
    expect(screen.getByText("Full Frame")).toBeInTheDocument();
    expect(screen.getByText("Vehicle Crop")).toBeInTheDocument();
    expect(screen.getAllByText("HR38AD4296").length).toBeGreaterThan(0);
    expect(screen.getByText("Licence Plate")).toBeInTheDocument();
    expect(screen.getByText((content) => content.includes("00:04.72"))).toBeInTheDocument();
    expect(screen.getByText("Technical Details")).toBeInTheDocument();
  });
});
