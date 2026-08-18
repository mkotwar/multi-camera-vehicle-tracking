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
  roiPreviewDraftUrl: vi.fn(),
  uploadConfigVideoSource: vi.fn(),
}));

vi.mock("../api/configs", () => mocks);

const baseConfig = {
  input: {
    cameras: [
      {
        camera_id: "CAM_001",
        source_type: "video",
        source: "D:/video.mp4",
        enabled: true,
        tracking_roi: {
          enabled: true,
          mode: "rectangle",
          rectangle: { x_min_fraction: 0, y_min_fraction: 0.4, x_max_fraction: 1, y_max_fraction: 0.75 },
          anchor: "bottom_center",
        },
      },
    ],
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
    mocks.roiPreviewDraftUrl.mockReturnValue("/api/configs/validation_rectangle_roi.yaml/roi-preview");
    mocks.uploadConfigVideoSource.mockResolvedValue({
      camera_id: "CAM_001",
      filename: "CAM_001_uploaded.mp4",
      source_path: "D:/project/multi-camera-vehicle-tracking/data/uploads/config_videos/CAM_001_uploaded.mp4",
      stored_path: "data/uploads/config_videos/CAM_001_uploaded.mp4",
    });
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
    await waitFor(() => expect(screen.getByLabelText("CAM_001 x min")).toHaveValue(0.1));

    fireEvent.click(screen.getAllByRole("button", { name: "Save" })[0]);
    await waitFor(() => expect(mocks.saveConfig).toHaveBeenCalled());
    const savedConfig = mocks.saveConfig.mock.calls[0][1];
    expect(savedConfig.input.cameras[0].tracking_roi.rectangle).toEqual({
      x_min_fraction: 0.1,
      y_min_fraction: 0.4,
      x_max_fraction: 0.9,
      y_max_fraction: 0.75,
    });

    fireEvent.click(screen.getByRole("button", { name: "Full Frame" }));
    fireEvent.click(screen.getAllByRole("button", { name: "Save" })[0]);
    await waitFor(() => expect(mocks.saveConfig).toHaveBeenCalledTimes(2));
    expect(mocks.saveConfig.mock.calls[1][1].input.cameras[0].tracking_roi.rectangle).toEqual({
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

  it("renders all cameras returned by the API", async () => {
    mocks.getConfig.mockResolvedValueOnce({
      config_name: "validation_rectangle_roi.yaml",
      path: "config/validation_rectangle_roi.yaml",
      config: {
        ...structuredClone(baseConfig),
        input: {
          cameras: [
            { camera_id: "CAM_001", source_type: "video", source: "D:/cam1.mp4", enabled: true },
            { camera_id: "CAM_002", source_type: "video", source: "D:/cam2.mp4", enabled: true },
            { camera_id: "CAM_003", source_type: "video", source: "D:/cam3.mp4", enabled: false },
          ],
        },
      },
      yaml_text: "",
      validation: { valid: true, errors: [], warnings: [] },
      inventory: [],
    });

    render(<MemoryRouter><SettingsPage /></MemoryRouter>);

    expect(await screen.findByDisplayValue("D:/cam1.mp4")).toBeInTheDocument();
    expect(screen.getByDisplayValue("D:/cam2.mp4")).toBeInTheDocument();
    expect(screen.getByDisplayValue("D:/cam3.mp4")).toBeInTheDocument();
  });

  it("adds and removes cameras without altering the others", async () => {
    render(<MemoryRouter><SettingsPage /></MemoryRouter>);

    await screen.findByDisplayValue("CAM_001");
    fireEvent.click(screen.getByRole("button", { name: "Add Camera" }));
    expect(screen.getByDisplayValue("CAM_002")).toBeInTheDocument();
    fireEvent.change(screen.getByDisplayValue("CAM_002"), { target: { value: "CAM_099" } });
    fireEvent.click(screen.getAllByRole("button", { name: "Remove" })[1]);
    fireEvent.click(screen.getAllByRole("button", { name: "Save" })[0]);

    await waitFor(() => expect(mocks.saveConfig).toHaveBeenCalled());
    const savedConfig = mocks.saveConfig.mock.calls[0][1];
    expect(savedConfig.input.cameras).toHaveLength(1);
    expect(savedConfig.input.cameras[0].camera_id).toBe("CAM_001");
    expect(savedConfig.input.cameras[0].source).toBe("D:/video.mp4");
  });

  it("keeps ROI edits isolated to the selected camera", async () => {
    mocks.getConfig.mockResolvedValueOnce({
      config_name: "validation_rectangle_roi.yaml",
      path: "config/validation_rectangle_roi.yaml",
      config: {
        ...structuredClone(baseConfig),
        input: {
          cameras: [
            structuredClone(baseConfig.input.cameras[0]),
            {
              camera_id: "CAM_002",
              source_type: "video",
              source: "D:/cam2.mp4",
              enabled: true,
              tracking_roi: {
                enabled: false,
                mode: "rectangle",
                rectangle: { x_min_fraction: 0.3, y_min_fraction: 0.2, x_max_fraction: 0.75, y_max_fraction: 0.9 },
                anchor: "bottom_center",
              },
            },
          ],
        },
      },
      yaml_text: "",
      validation: { valid: true, errors: [], warnings: [] },
      inventory: [],
    });
    render(<MemoryRouter><SettingsPage /></MemoryRouter>);

    fireEvent.change(await screen.findByLabelText("CAM_001 x min"), { target: { value: "0.12" } });
    fireEvent.click(screen.getAllByRole("button", { name: "Save" })[0]);

    await waitFor(() => expect(mocks.saveConfig).toHaveBeenCalled());
    const savedConfig = mocks.saveConfig.mock.calls[0][1];
    expect(savedConfig.input.cameras[0].tracking_roi.rectangle.x_min_fraction).toBe(0.12);
    expect(savedConfig.input.cameras[1].tracking_roi.rectangle.x_min_fraction).toBe(0.3);
    expect(savedConfig.input.cameras[1].tracking_roi.enabled).toBe(false);
  });

  it("keeps CAM_003 ROI edits isolated from CAM_001 and CAM_002", async () => {
    mocks.getConfig.mockResolvedValueOnce({
      config_name: "validation_rectangle_roi.yaml",
      path: "config/validation_rectangle_roi.yaml",
      config: {
        ...structuredClone(baseConfig),
        input: {
          cameras: [
            structuredClone(baseConfig.input.cameras[0]),
            {
              ...structuredClone(baseConfig.input.cameras[0]),
              camera_id: "CAM_002",
              source: "D:/cam2.mp4",
              tracking_roi: {
                enabled: true,
                mode: "rectangle",
                rectangle: { x_min_fraction: 0.2, y_min_fraction: 0.2, x_max_fraction: 0.8, y_max_fraction: 0.8 },
                anchor: "bottom_center",
              },
            },
            {
              ...structuredClone(baseConfig.input.cameras[0]),
              camera_id: "CAM_003",
              source: "D:/cam3.mp4",
              tracking_roi: {
                enabled: true,
                mode: "rectangle",
                rectangle: { x_min_fraction: 0.3, y_min_fraction: 0.3, x_max_fraction: 0.9, y_max_fraction: 0.9 },
                anchor: "bottom_center",
              },
            },
          ],
        },
      },
      yaml_text: "",
      validation: { valid: true, errors: [], warnings: [] },
      inventory: [],
    });
    render(<MemoryRouter><SettingsPage /></MemoryRouter>);

    fireEvent.change(await screen.findByLabelText("CAM_003 x min"), { target: { value: "0.35" } });
    fireEvent.click(screen.getAllByRole("button", { name: "Save" })[0]);

    await waitFor(() => expect(mocks.saveConfig).toHaveBeenCalled());
    const savedConfig = mocks.saveConfig.mock.calls[0][1];
    expect(savedConfig.input.cameras[0].tracking_roi.rectangle.x_min_fraction).toBe(0);
    expect(savedConfig.input.cameras[1].tracking_roi.rectangle.x_min_fraction).toBe(0.2);
    expect(savedConfig.input.cameras[2].tracking_roi.rectangle.x_min_fraction).toBe(0.35);
  });

  it("loads preview frames with newly added unsaved camera details", async () => {
    mocks.getConfig.mockResolvedValueOnce({
      config_name: "validation_rectangle_roi.yaml",
      path: "config/validation_rectangle_roi.yaml",
      config: {
        ...structuredClone(baseConfig),
        input: {
          cameras: [
            structuredClone(baseConfig.input.cameras[0]),
            { ...structuredClone(baseConfig.input.cameras[0]), camera_id: "CAM_002", source: "D:/cam2.mp4" },
          ],
        },
      },
      yaml_text: "",
      validation: { valid: true, errors: [], warnings: [] },
      inventory: [],
    });
    render(<MemoryRouter><SettingsPage /></MemoryRouter>);

    await screen.findByDisplayValue("CAM_001");
    fireEvent.click(screen.getByRole("button", { name: "Add Camera" }));
    fireEvent.change(screen.getAllByPlaceholderText("Stored video path or existing absolute path")[2], { target: { value: "D:/cam3.mp4" } });
    fireEvent.click(screen.getAllByRole("button", { name: "Load frame" })[2]);

    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
    const [_url, options] = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(options.method).toBe("POST");
    expect(JSON.parse(options.body)).toEqual({
      camera: expect.objectContaining({
        camera_id: "CAM_003",
        source_type: "video",
        source: "D:/cam3.mp4",
      }),
    });
  });

  it("loads preview with the edited source before save", async () => {
    render(<MemoryRouter><SettingsPage /></MemoryRouter>);

    const source = await screen.findByDisplayValue("D:/video.mp4");
    fireEvent.change(source, { target: { value: "D:/new-source.mp4" } });
    fireEvent.click(screen.getByRole("button", { name: "Load frame" }));

    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
    const [_url, options] = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(JSON.parse(options.body).camera.source).toBe("D:/new-source.mp4");
  });

  it("uploads a selected video and updates the camera source field", async () => {
    render(<MemoryRouter><SettingsPage /></MemoryRouter>);

    const file = new File(["video"], "new-camera.mp4", { type: "video/mp4" });
    fireEvent.change(await screen.findByLabelText("Upload video for CAM_001"), {
      target: { files: [file] },
    });

    await waitFor(() => expect(mocks.uploadConfigVideoSource).toHaveBeenCalledWith("validation_rectangle_roi.yaml", "CAM_001", file));
    expect(await screen.findByDisplayValue("D:/project/multi-camera-vehicle-tracking/data/uploads/config_videos/CAM_001_uploaded.mp4")).toBeInTheDocument();
    expect(screen.getByText("Unsaved changes")).toBeInTheDocument();
  });

  it("accepts drag and drop uploads for camera videos", async () => {
    render(<MemoryRouter><SettingsPage /></MemoryRouter>);

    const file = new File(["video"], "drop-source.mp4", { type: "video/mp4" });
    const dropzone = await screen.findByLabelText("Drop video for CAM_001");
    fireEvent.dragEnter(dropzone, { dataTransfer: { files: [file] } });
    fireEvent.drop(dropzone, { dataTransfer: { files: [file] } });

    await waitFor(() => expect(mocks.uploadConfigVideoSource).toHaveBeenCalledWith("validation_rectangle_roi.yaml", "CAM_001", file));
  });

  it("keeps the previous source and shows a friendly error when upload fails", async () => {
    mocks.uploadConfigVideoSource.mockRejectedValueOnce({ status: 405, detail: { detail: "Method Not Allowed" } });
    render(<MemoryRouter><SettingsPage /></MemoryRouter>);

    const file = new File(["video"], "bad-camera.mp4", { type: "video/mp4" });
    fireEvent.change(await screen.findByLabelText("Upload video for CAM_001"), {
      target: { files: [file] },
    });

    expect(await screen.findByDisplayValue("D:/video.mp4")).toBeInTheDocument();
    expect(screen.getByText("Video upload failed. The server rejected the upload request (HTTP 405). Method Not Allowed")).toBeInTheDocument();
  });
});
