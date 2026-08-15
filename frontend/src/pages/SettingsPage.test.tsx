import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SettingsPage } from "./SettingsPage";

const mocks = vi.hoisted(() => ({
  listConfigs: vi.fn(),
  getConfig: vi.fn(),
  validateConfig: vi.fn(),
  saveConfig: vi.fn(),
  cloneConfig: vi.fn(),
  roiPreviewUrl: vi.fn(),
}));

vi.mock("../api/configs", () => mocks);

const baseConfig = {
  input: {
    cameras: [{ camera_id: "CAM_001", source_type: "video", source: "D:/video.mp4", enabled: true }],
  },
  detection: { backend: "ocr_mukul", model_path: "D:/model.pt", confidence_threshold: 0.2, iou_threshold: 0.45, image_size: 1024, agnostic_nms: false },
  tracking: { backend: "ocr_mukul_supervision_bytetrack", track_activation_threshold: 0.25, lost_track_buffer: 150, minimum_matching_threshold: 0.7, minimum_consecutive_frames: 3 },
  tracking_roi: {
    enabled: true,
    mode: "rectangle",
    rectangle: { x_min_fraction: 0, y_min_fraction: 0.4, x_max_fraction: 1, y_max_fraction: 0.75 },
    anchor: "bottom_center",
  },
  evidence: { enabled: true, maximum_candidates_per_track: 7 },
  vehicle_enrichment: {
    enabled: true,
    best_crops_per_track: 3,
    vehicle_attributes: { colour: { enabled: true }, body_type: { enabled: false } },
    plate: { detection_enabled: true, recognition_enabled: true, detector: { enabled: true, model_path: "D:/plate.pt" }, ocr: { enabled: true, backend: "ocr_mukul_adapter" } },
  },
  vehicle_identity: {
    enabled: true,
    conservative: { enabled: true, acceptance_threshold: 0.7, ambiguity_margin: 0.03, vehicle_consistency_floor: 0.58 },
    plate_assistance: { enabled: true, contradiction_veto: true },
    stationary_recovery: { enabled: false },
  },
  output: { root_directory: "outputs/runs", save_run_config: true },
  ingestion: { worker_count: 7, frame_queue_size: 200 },
};

describe("SettingsPage", () => {
  afterEach(() => cleanup());

  beforeEach(() => {
    vi.clearAllMocks();
    window.confirm = vi.fn(() => true);
    globalThis.fetch = vi.fn(async () => new Response(new Blob(["preview"], { type: "image/jpeg" }), { status: 200 }));
    URL.createObjectURL = vi.fn(() => "blob:roi-preview");
    URL.revokeObjectURL = vi.fn();
    mocks.listConfigs.mockResolvedValue([{ config_name: "validation_rectangle_roi.yaml", path: "config/validation_rectangle_roi.yaml", production: false, preferred: true }]);
    mocks.getConfig.mockResolvedValue({
      config_name: "validation_rectangle_roi.yaml",
      path: "config/validation_rectangle_roi.yaml",
      config: structuredClone(baseConfig),
      yaml_text: "tracking_roi:\n  enabled: true\n",
      validation: { valid: true, errors: [], warnings: [] },
      inventory: [{ path: "tracking.minimum_matching_threshold", type: "float", default: 0.7, required: true, restart_required: true, operator_level: "advanced", runtime_effect: "Tracking" }],
    });
    mocks.validateConfig.mockResolvedValue({ valid: true, errors: [], warnings: [] });
    mocks.saveConfig.mockResolvedValue({ valid: true, errors: [], warnings: [], saved_path: "config/validation_rectangle_roi.yaml", config_name: "validation_rectangle_roi.yaml", yaml_text: "saved: true\n" });
    mocks.roiPreviewUrl.mockReturnValue("/api/configs/validation_rectangle_roi.yaml/roi-preview?camera_id=CAM_001");
  });

  it("loads config values into structured controls and saves numeric edits", async () => {
    render(<MemoryRouter><SettingsPage /></MemoryRouter>);

    expect(await screen.findByDisplayValue("0.2")).toBeInTheDocument();
    const lostBuffer = screen.getByLabelText("Lost track buffer");
    fireEvent.change(lostBuffer, { target: { value: "160" } });
    fireEvent.click(screen.getAllByRole("button", { name: "Save" })[0]);

    await waitFor(() => expect(mocks.saveConfig).toHaveBeenCalled());
    expect(mocks.saveConfig.mock.calls[0][1].tracking.lost_track_buffer).toBe(160);
  });

  it("draws ROI using normalized image coordinates and supports full-frame reset", async () => {
    render(<MemoryRouter><SettingsPage /></MemoryRouter>);

    fireEvent.click(await screen.findByRole("button", { name: "Load frame" }));
    const image = await screen.findByAltText("ROI preview frame");
    const boundsSpy = vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({
      left: 0,
      top: 0,
      width: 1000,
      height: 500,
      right: 1000,
      bottom: 500,
      x: 0,
      y: 0,
      toJSON: () => {},
    } as DOMRect);
    const stage = image.closest(".roi-stage") as HTMLElement;
    fireEvent(stage, new MouseEvent("pointerdown", { bubbles: true, clientX: 100, clientY: 200 }));
    fireEvent(stage, new MouseEvent("pointermove", { bubbles: true, clientX: 900, clientY: 375 }));
    fireEvent(stage, new MouseEvent("pointerup", { bubbles: true }));
    await waitFor(() => expect(screen.getByLabelText("x min")).toHaveValue(0.1));

    fireEvent.click(screen.getAllByRole("button", { name: "Save" })[0]);
    await waitFor(() => expect(mocks.saveConfig).toHaveBeenCalled());
    const savedConfig = mocks.saveConfig.mock.calls[0][1];
    expect(savedConfig.tracking_roi.rectangle).toEqual({
      x_min_fraction: 0.1,
      y_min_fraction: 0.4,
      x_max_fraction: 0.9,
      y_max_fraction: 0.75,
    });

    fireEvent.click(screen.getByRole("button", { name: "Full Frame" }));
    fireEvent.click(screen.getAllByRole("button", { name: "Save" })[0]);
    await waitFor(() => expect(mocks.saveConfig).toHaveBeenCalledTimes(2));
    expect(mocks.saveConfig.mock.calls[1][1].tracking_roi.rectangle).toEqual({
      x_min_fraction: 0,
      y_min_fraction: 0,
      x_max_fraction: 1,
      y_max_fraction: 1,
    });
    boundsSpy.mockRestore();
  });

  it("renders validation errors with field paths", async () => {
    mocks.validateConfig.mockResolvedValue({
      valid: false,
      errors: [{ rule: "roi.y_order", path: "tracking_roi.rectangle.y_min_fraction", message: "Must be less than y_max_fraction." }],
      warnings: [],
    });
    render(<MemoryRouter><SettingsPage /></MemoryRouter>);

    await screen.findByText("Pipeline Configuration");
    fireEvent.click(screen.getAllByRole("button", { name: "Validate" })[0]);

    expect(await screen.findByText("tracking_roi.rectangle.y_min_fraction")).toBeInTheDocument();
    expect(screen.getByText(/Must be less than y_max_fraction/)).toBeInTheDocument();
  });
});
