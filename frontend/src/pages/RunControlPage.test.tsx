import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RunControlPage } from "./RunControlPage";

const mocks = vi.hoisted(() => ({
  listConfigs: vi.fn(),
  fetchPipelineRunJobs: vi.fn(),
  fetchPipelineRunJob: vi.fn(),
  fetchPipelineRunLogs: vi.fn(),
  fetchPipelineLaunchSummary: vi.fn(),
  fetchDbAutoImportSetting: vi.fn(),
  updateDbAutoImportSetting: vi.fn(),
  startPipelineRun: vi.fn(),
  cancelPipelineRun: vi.fn(),
}));

vi.mock("../api/configs", () => ({ listConfigs: mocks.listConfigs }));
vi.mock("../api/pipelineRuns", () => ({
  fetchPipelineRunJobs: mocks.fetchPipelineRunJobs,
  fetchPipelineRunJob: mocks.fetchPipelineRunJob,
  fetchPipelineRunLogs: mocks.fetchPipelineRunLogs,
  fetchPipelineLaunchSummary: mocks.fetchPipelineLaunchSummary,
  fetchDbAutoImportSetting: mocks.fetchDbAutoImportSetting,
  updateDbAutoImportSetting: mocks.updateDbAutoImportSetting,
  startPipelineRun: mocks.startPipelineRun,
  cancelPipelineRun: mocks.cancelPipelineRun,
}));

const runningJob = {
  job_id: "JOB_TEST",
  config_name: "validation_rectangle_roi.yaml",
  status: "RUNNING",
  created_at: "2026-08-15T10:00:00+00:00",
  started_at: "2026-08-15T10:00:01+00:00",
  pid: 1234,
  run_id: null,
  current_stage: "DETECTION_TRACKING",
  processed_frames: 42,
  elapsed_seconds: 8,
};

const completedJob = {
  ...runningJob,
  status: "COMPLETED",
  current_stage: "COMPLETED",
  run_id: "20260815_155243",
  finished_at: "2026-08-15T10:02:00+00:00",
};

describe("RunControlPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.listConfigs.mockResolvedValue([{ config_name: "validation_rectangle_roi.yaml", path: "config/validation_rectangle_roi.yaml", production: false, preferred: true }]);
    mocks.fetchPipelineRunJobs.mockResolvedValue([]);
    mocks.fetchPipelineLaunchSummary.mockResolvedValue({
      config_name: "validation_rectangle_roi.yaml",
      valid: true,
      errors: [],
      input_sources: [{ camera_id: "CAM_001", source_type: "video", source: "D:/video.mp4" }],
      tracking_roi: { enabled: true, mode: "rectangle", rectangle: {}, anchor: "bottom_center" },
      plate_ocr_enabled: false,
      plate_detector_enabled: false,
      physical_identity_enabled: true,
      stationary_recovery_enabled: false,
      db_import_after_run: true,
    });
    mocks.fetchPipelineRunLogs.mockResolvedValue({ job_id: "JOB_TEST", log_file: "pipeline.log", lines: ["Pipeline started"], limit: 180 });
    mocks.fetchDbAutoImportSetting.mockResolvedValue({ key: "DB_IMPORT_AFTER_RUN", enabled: true, configured_value: "true", source: ".env" });
    mocks.updateDbAutoImportSetting.mockResolvedValue({ key: "DB_IMPORT_AFTER_RUN", enabled: false, configured_value: "false", source: ".env" });
    mocks.startPipelineRun.mockResolvedValue(runningJob);
    mocks.fetchPipelineRunJob.mockResolvedValue(runningJob);
    mocks.cancelPipelineRun.mockResolvedValue({ ...runningJob, status: "CANCELLED", current_stage: "CANCELLED" });
  });

  afterEach(() => cleanup());

  it("loads config summary and starts after confirmation", async () => {
    render(<MemoryRouter><RunControlPage /></MemoryRouter>);

    expect(await screen.findByText("Pipeline Run Control")).toBeInTheDocument();
    expect(await screen.findByText("DB auto-import")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Start Run" }));
    expect(await screen.findByText("Start pipeline run?")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Confirm Start" }));

    await waitFor(() => expect(mocks.startPipelineRun).toHaveBeenCalledWith("validation_rectangle_roi.yaml"));
    expect(await screen.findByText("JOB_TEST")).toBeInTheDocument();
  });

  it("toggles DB auto-import from run control", async () => {
    render(<MemoryRouter><RunControlPage /></MemoryRouter>);

    const toggle = await screen.findByRole("checkbox", { name: /DB auto-import/i });
    expect(toggle).toBeChecked();
    fireEvent.click(toggle);

    await waitFor(() => expect(mocks.updateDbAutoImportSetting).toHaveBeenCalledWith(false));
    expect(await screen.findByText("Off")).toBeInTheDocument();
  });

  it("disables start when selected config is invalid", async () => {
    mocks.fetchPipelineLaunchSummary.mockResolvedValue({
      config_name: "validation_rectangle_roi.yaml",
      valid: false,
      errors: [{ path: "tracking_roi.rectangle.y_min_fraction", message: "Must be less than y_max_fraction." }],
      input_sources: [],
      tracking_roi: { enabled: false },
      plate_ocr_enabled: false,
      plate_detector_enabled: false,
      physical_identity_enabled: false,
      stationary_recovery_enabled: false,
      db_import_after_run: false,
    });
    render(<MemoryRouter><RunControlPage /></MemoryRouter>);

    expect(await screen.findByText("tracking_roi.rectangle.y_min_fraction")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start Run" })).toBeDisabled();
  });

  it("renders running logs and cancels", async () => {
    mocks.fetchPipelineRunJobs.mockResolvedValue([runningJob]);
    render(<MemoryRouter><RunControlPage /></MemoryRouter>);

    expect(await screen.findByText("Pipeline started")).toBeInTheDocument();
    expect(screen.getByText("DETECTION_TRACKING")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Cancel Run" }));

    await waitFor(() => expect(mocks.cancelPipelineRun).toHaveBeenCalledWith("JOB_TEST"));
  });

  it("shows completed links with run_id and failed error messages", async () => {
    mocks.fetchPipelineRunJobs.mockResolvedValue([completedJob]);
    mocks.fetchPipelineRunJob.mockResolvedValue(completedJob);
    render(<MemoryRouter><RunControlPage /></MemoryRouter>);

    expect(await screen.findByRole("link", { name: "Open Vehicles" })).toHaveAttribute("href", "/vehicles?run_id=20260815_155243");
    expect(screen.getByRole("link", { name: "Open Video Chat" })).toHaveAttribute("href", "/video-chat?run_id=20260815_155243");

    cleanup();
    mocks.fetchPipelineRunJobs.mockResolvedValue([{ ...runningJob, status: "FAILED", error_message: "Pipeline failed" }]);
    mocks.fetchPipelineRunJob.mockResolvedValue({ ...runningJob, status: "FAILED", error_message: "Pipeline failed" });
    render(<MemoryRouter><RunControlPage /></MemoryRouter>);
    expect(await screen.findByText("Pipeline failed")).toBeInTheDocument();
  });
});
