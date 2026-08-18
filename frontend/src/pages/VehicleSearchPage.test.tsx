import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { VehicleSearchPage } from "./VehicleSearchPage";

const { fetchExperimentalVehicles, fetchFilterOptions, fetchPlateAssistedVehicles, fetchStationaryRecoveredVehicles, fetchTrackReconciliation, fetchTracks, fetchVehicles, searchVehicles } = vi.hoisted(() => ({
  fetchExperimentalVehicles: vi.fn(),
  fetchFilterOptions: vi.fn(),
  fetchPlateAssistedVehicles: vi.fn(),
  fetchStationaryRecoveredVehicles: vi.fn(),
  fetchTrackReconciliation: vi.fn(),
  fetchTracks: vi.fn(),
  fetchVehicles: vi.fn(),
  searchVehicles: vi.fn(),
}));

vi.mock("../api/filters", () => ({
  fetchFilterOptions,
}));

vi.mock("../api/runs", () => ({
  fetchExperimentalVehicles,
  fetchPlateAssistedVehicles,
  fetchStationaryRecoveredVehicles,
  fetchTrackReconciliation,
}));

vi.mock("../api/tracks", () => ({
  fetchTracks,
  fetchVehicles,
}));

vi.mock("../api/vehicleSearch", () => ({
  searchVehicles,
}));

vi.mock("../api/client", () => ({
  ApiError: class ApiError extends Error {
    status: number;
    detail: unknown;

    constructor(status: number, detail: unknown) {
      super(`Request failed: ${status}`);
      this.status = status;
      this.detail = detail;
    }
  },
}));

describe("VehicleSearchPage", () => {
  afterEach(() => {
    cleanup();
  });

  beforeEach(() => {
    fetchFilterOptions.mockReset();
    fetchExperimentalVehicles.mockReset();
    fetchPlateAssistedVehicles.mockReset();
    fetchStationaryRecoveredVehicles.mockReset();
    fetchTrackReconciliation.mockReset();
    fetchTracks.mockReset();
    fetchVehicles.mockReset();
    searchVehicles.mockReset();
    fetchFilterOptions.mockResolvedValue({
      runs: ["latest", "20260808_182124"],
      cameras: ["CAM_001"],
      vehicle_classes: ["3WHEELER", "BUS", "CAR", "MOTORCYCLE", "TRUCK"],
      colours: ["BLACK", "WHITE", "GREY", "SILVER", "RED", "PINK", "BLUE", "GREEN", "YELLOW", "ORANGE", "BROWN", "BEIGE", "PURPLE", "OTHER", "UNKNOWN"],
    });
    fetchTracks.mockResolvedValue([
      {
        run_id: "20260808_182124",
        camera_id: "CAM_001",
        track_id: "TRACK_1",
        local_track_id: "CAM_001:TRACK_1",
        vehicle_class: "CAR",
        colour: "WHITE",
        plate_text: "DL8CAF5030",
        plate_detected: true,
        plate_text_confidence: 0.86,
        first_seen_seconds: 1.2,
        last_seen_seconds: 2.8,
        duration_seconds: 1.6,
        status: "COMPLETED",
      },
    ]);
    fetchVehicles.mockResolvedValue([
      {
        run_id: "20260808_182124",
        vehicle_id: "VEHICLE_004",
        vehicle_key: "VEHICLE_004",
        primary_camera_id: "CAM_001",
        camera_ids: ["CAM_001"],
        vehicle_class: "CAR",
        vehicle_colour: "WHITE",
        first_seen_seconds: 1.2,
        last_seen_seconds: 6.8,
        identity_confidence: 0.96,
        identity_method: "plate_assisted",
        identity_status: "MERGED",
        consensus_plate_text: "HR38AD4296",
        plate_confidence: 0.75,
        member_track_ids: ["CAM_001:TRACK_4", "CAM_001:TRACK_14"],
        member_track_count: 2,
      },
    ]);
    searchVehicles.mockResolvedValue({
      run_id: "20260808_182124",
      original_query: "How many white cars are there?",
      parsed_query: {
        intent: "COUNT",
        vehicle_class: "CAR",
        colour: "WHITE",
        start_time: null,
        end_time: null,
        camera_id: null,
      },
      analytics_result: {
        total: 6,
        by_class: { CAR: 6 },
        by_colour: { WHITE: 6 },
        vehicle_ids: ["CAM_001:TRACK_13", "CAM_001:TRACK_19"],
      },
      response: "There are 6 white cars.",
    });
    fetchTrackReconciliation.mockResolvedValue({
      run_id: "20260808_182124",
      available: true,
      message: null,
      metrics: {
        raw_bytetrack_unique_tracks: 125,
        reconciled_vehicle_identities: 120,
        track_fragments_merged: 5,
        accepted_matches: 5,
        ambiguous_matches: 0,
      },
      config: {},
      tracks: [
        {
          local_track_id: "CAM_001:TRACK_44",
          track_id: "TRACK_44",
          camera_id: "CAM_001",
          status: "COMPLETED",
          vehicle_id: "VEHICLE_037",
          final_class: "car",
          first_frame: 489,
          last_frame: 506,
          first_timestamp_seconds: 16.3,
          last_timestamp_seconds: 16.86,
          vehicle_enrichment: { vehicle_class: "CAR", vehicle_colour: { label: "WHITE", status: "completed" } },
          reconciliation: { matched: false, result: "unmatched", reason: "no_recent_candidate" },
        },
        {
          local_track_id: "CAM_001:TRACK_48",
          track_id: "TRACK_48",
          camera_id: "CAM_001",
          status: "COMPLETED",
          vehicle_id: "VEHICLE_037",
          final_class: "car",
          first_frame: 534,
          last_frame: 565,
          first_timestamp_seconds: 17.8,
          last_timestamp_seconds: 18.83,
          vehicle_enrichment: { vehicle_class: "CAR", vehicle_colour: { label: "WHITE", status: "completed" } },
          reconciliation: {
            matched: true,
            previous_track_id: "CAM_001:TRACK_44",
            score: 0.747848,
            second_best_score: 0,
            time_gap_frames: 28,
            time_gap_seconds: 0.9333333333333336,
            result: "accepted",
          },
        },
      ],
      accepted_associations: [
        {
          old_track: "CAM_001:TRACK_44",
          new_track: "CAM_001:TRACK_48",
          vehicle_id: "VEHICLE_037",
          gap_frames: 28,
          gap_seconds: 0.9333333333333336,
          score: 0.747848,
          second_best_score: 0,
          colour: "WHITE",
          class: "CAR",
          result: "ACCEPTED",
        },
      ],
      manual_validation: [],
      visual_evidence: [],
      paths: {},
    });
    fetchExperimentalVehicles.mockResolvedValue({
      run_id: "20260808_182124",
      experimental: true,
      available: false,
      message: "Persistent vehicle identity experiment has not been run for this run.",
      metrics: {},
      analytics_simulation: {},
      existing_reconciliation_baseline: {},
      config: {},
      calibration: {},
      vehicles: [],
      vehicle_id_map: {},
      association_decisions: [],
      paths: {},
    });
    fetchStationaryRecoveredVehicles.mockResolvedValue({
      run_id: "20260808_182124",
      experimental: true,
      stage: "stationary_recovery",
      available: false,
      message: "Stationary recovery experiment has not been run for this run.",
      metrics: {},
      analytics_simulation: {},
      config: {},
      calibration: {},
      persistent_vehicles: [],
      persistent_vehicle_id_map: {},
      recovery_decisions: [],
      recovery_scores: [],
      paths: {},
    });
    fetchPlateAssistedVehicles.mockResolvedValue({
      run_id: "20260808_182124",
      experimental: true,
      stage: "plate_assisted_identity",
      available: true,
      message: null,
      verification: {},
      plate_coverage: {
        completed_tracks: 23,
        plate_detected_count: 11,
        readable_plate_count: 11,
        high_quality_plate_count: 11,
        exact_matching_plate_pairs: 3,
      },
      baseline_without_plate: { reconciled_identities: 16 },
      plate_assisted: {
        raw_completed_tracks: 23,
        reconciled_identities: 13,
        duplicates_removed: 10,
        true_fragment_merges: 3,
        false_merges: 0,
      },
      vehicles: [
        {
          vehicle_id: "VEHICLE_004",
          camera_id: "CAM_001",
          member_tracks: ["CAM_001:TRACK_4", "CAM_001:TRACK_14"],
          member_track_ids: ["CAM_001:TRACK_4", "CAM_001:TRACK_14"],
          final_class: "CAR",
          first_seen_seconds: 0.06,
          last_seen_seconds: 6.7,
          contact_sheet_url: "/api/media/plate_assisted_contact_sheets/20260808_182124/same__TRACK_4__TRACK_14.jpg",
          plate: {
            consensus_text: "HR38AD4296",
            quality: "HIGH",
            status: "EXACT / HIGH QUALITY",
            member_plates: [
              {
                local_track_id: "CAM_001:TRACK_4",
                normalized_plate_text: "HR38AD4296",
                quality: "HIGH",
                plate_evidence_status: "EXACT / HIGH QUALITY",
                plate_detection_confidence: 0.744,
                plate_text_confidence: 0.744,
                plate_crop_url: "/api/media/florence_selected_crops/20260808_182124/CAM_001/TRACK_4/plate/frame.jpg",
                vehicle_crop_url: "/api/media/florence_selected_crops/20260808_182124/CAM_001/TRACK_4/frame.jpg",
                plate_ocr_reason: "ocr_completed",
              },
              {
                local_track_id: "CAM_001:TRACK_14",
                normalized_plate_text: "HR38AD4296",
                quality: "HIGH",
                plate_evidence_status: "EXACT / HIGH QUALITY",
                plate_detection_confidence: 0.755,
                plate_text_confidence: 0.755,
                plate_ocr_reason: "ocr_completed",
              },
            ],
          },
          association_reasons: ["PLATE_EXACT_MATCH | SPATIAL_MATCH"],
        },
        {
          vehicle_id: "VEHICLE_010",
          camera_id: "CAM_001",
          member_tracks: ["CAM_001:TRACK_16"],
          member_track_ids: ["CAM_001:TRACK_16"],
          final_class: "BUS",
          plate: {
            consensus_text: "HR30T42",
            quality: "HIGH",
            status: "PARTIAL",
            member_plates: [
              {
                local_track_id: "CAM_001:TRACK_16",
                normalized_plate_text: "HR30T42",
                quality: "HIGH",
                plate_evidence_status: "PARTIAL",
                plate_detection_confidence: 0.821,
                plate_text_confidence: 0.821,
                plate_ocr_reason: "ocr_completed",
              },
            ],
          },
          association_reasons: [],
        },
      ],
      vehicle_id_map: { "CAM_001:TRACK_4": "VEHICLE_004", "CAM_001:TRACK_14": "VEHICLE_004" },
      track_plate_consensus: [],
      association_decisions: [],
      plate_pair_scores: [],
      identity_scores: [],
      paths: {},
    });
  });

  it("renders full supported class and colour options", async () => {
    render(
      <MemoryRouter initialEntries={["/vehicles"]}>
        <VehicleSearchPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchFilterOptions).toHaveBeenCalled());

    expect(screen.getByRole("option", { name: "BUS" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "TRUCK" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "BLACK" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "BLUE" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "OTHER" })).toBeInTheDocument();
  });

  it("submits selected class and colour filter values unchanged", async () => {
    render(
      <MemoryRouter initialEntries={["/vehicles"]}>
        <VehicleSearchPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchTracks).toHaveBeenCalled());
    fetchTracks.mockClear();

    fireEvent.change(screen.getByLabelText("Class"), { target: { value: "BUS" } });
    fireEvent.change(screen.getByLabelText("Colour"), { target: { value: "BLUE" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));

    await waitFor(() => expect(fetchTracks).toHaveBeenCalledTimes(1));
    const query = fetchTracks.mock.calls[0][0] as URLSearchParams;

    expect(query.get("vehicle_class")).toBe("BUS");
    expect(query.get("colour")).toBe("BLUE");
  });

  it("sends the natural-language request with the selected run id", async () => {
    render(
      <MemoryRouter initialEntries={["/vehicles"]}>
        <VehicleSearchPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchTracks).toHaveBeenCalled());

    fireEvent.change(screen.getByLabelText("Run"), { target: { value: "20260808_182124" } });
    fireEvent.change(screen.getByLabelText("Natural-language vehicle search"), { target: { value: "How many white cars are there?" } });
    fireEvent.click(screen.getByRole("button", { name: "Run natural-language vehicle search" }));

    await waitFor(() => expect(searchVehicles).toHaveBeenCalledTimes(1));
    expect(searchVehicles).toHaveBeenCalledWith({
      query: "How many white cars are there?",
      run_id: "20260808_182124",
    });
  });

  it("shows a loading state while natural-language search is in flight", async () => {
    let resolveSearch: (value: unknown) => void = () => undefined;
    searchVehicles.mockReturnValue(new Promise((resolve) => { resolveSearch = resolve; }));
    render(
      <MemoryRouter initialEntries={["/vehicles"]}>
        <VehicleSearchPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchTracks).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText("Natural-language vehicle search"), { target: { value: "How many vehicles are there?" } });
    fireEvent.click(screen.getByRole("button", { name: "Run natural-language vehicle search" }));

    expect(screen.getByRole("button", { name: "Run natural-language vehicle search" })).toBeDisabled();
    resolveSearch({
      run_id: "latest",
      original_query: "How many vehicles are there?",
      parsed_query: { intent: "COUNT", vehicle_class: null, colour: null, start_time: null, end_time: null, camera_id: null },
      analytics_result: { total: 41, vehicle_ids: [] },
      response: "There are 41 vehicles.",
    });
    await waitFor(() => expect(screen.getByText("There are 41 vehicles.")).toBeInTheDocument());
  });

  it("renders count search responses and parsed query details", async () => {
    render(
      <MemoryRouter initialEntries={["/vehicles"]}>
        <VehicleSearchPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchTracks).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText("Natural-language vehicle search"), { target: { value: "How many white cars are there?" } });
    fireEvent.submit(screen.getByLabelText("Natural-language vehicle search").closest("form")!);

    await waitFor(() => expect(screen.getByText("There are 6 white cars.")).toBeInTheDocument());
    expect(screen.getByText("Matches: 6")).toBeInTheDocument();
    expect(screen.getByText("CAM_001:TRACK_13")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Parsed query"));
    expect(screen.getByText("COUNT")).toBeInTheDocument();
    expect(screen.getAllByText("CAR").length).toBeGreaterThan(0);
  });

  it("renders list responses with vehicle ids", async () => {
    searchVehicles.mockResolvedValue({
      run_id: "20260808_182124",
      original_query: "Show black motorcycles between 5 and 10 seconds",
      parsed_query: { intent: "LIST", vehicle_class: "MOTORCYCLE", colour: "BLACK", start_time: 5, end_time: 10, camera_id: null },
      analytics_result: {
        total: 5,
        vehicle_ids: ["CAM_001:TRACK_14", "CAM_001:TRACK_18", "CAM_001:TRACK_17", "CAM_001:TRACK_25", "CAM_001:TRACK_23"],
      },
      response: "5 black motorcycles between 5.0 and 10.0 seconds were observed.",
    });
    render(
      <MemoryRouter initialEntries={["/vehicles"]}>
        <VehicleSearchPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchTracks).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText("Natural-language vehicle search"), { target: { value: "Show black motorcycles between 5 and 10 seconds" } });
    fireEvent.click(screen.getByRole("button", { name: "Run natural-language vehicle search" }));

    await waitFor(() => expect(screen.getByText("5 black motorcycles between 5.0 and 10.0 seconds were observed.")).toBeInTheDocument());
    expect(screen.getByText("CAM_001:TRACK_23")).toBeInTheDocument();
  });

  it("renders zero-result responses as a normal result", async () => {
    searchVehicles.mockResolvedValue({
      run_id: "20260808_182124",
      original_query: "Show white cars between 5 and 10 seconds",
      parsed_query: { intent: "LIST", vehicle_class: "CAR", colour: "WHITE", start_time: 5, end_time: 10, camera_id: null },
      analytics_result: { total: 0, vehicle_ids: [] },
      response: "No white cars were observed between 5.0 and 10.0 seconds.",
    });
    render(
      <MemoryRouter initialEntries={["/vehicles"]}>
        <VehicleSearchPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchTracks).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText("Natural-language vehicle search"), { target: { value: "Show white cars between 5 and 10 seconds" } });
    fireEvent.click(screen.getByRole("button", { name: "Run natural-language vehicle search" }));

    await waitFor(() => expect(screen.getByText("No white cars were observed between 5.0 and 10.0 seconds.")).toBeInTheDocument());
    expect(screen.getByText("Matches: 0")).toBeInTheDocument();
  });

  it("renders invalid query errors from the backend", async () => {
    searchVehicles.mockRejectedValue(new Error("I couldn't understand that query. Try specifying a vehicle type, colour, or time range."));
    render(
      <MemoryRouter initialEntries={["/vehicles"]}>
        <VehicleSearchPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchTracks).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText("Natural-language vehicle search"), { target: { value: "show dark vehicles" } });
    fireEvent.click(screen.getByRole("button", { name: "Run natural-language vehicle search" }));

    await waitFor(() => expect(screen.getByText("I couldn't understand that query. Try specifying a vehicle type, colour, or time range.")).toBeInTheDocument());
  });

  it("submits natural-language search with the Enter key", async () => {
    render(
      <MemoryRouter initialEntries={["/vehicles"]}>
        <VehicleSearchPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchTracks).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText("Natural-language vehicle search"), { target: { value: "How many vehicles are there?" } });
    fireEvent.keyDown(screen.getByLabelText("Natural-language vehicle search"), { key: "Enter", code: "Enter" });
    fireEvent.submit(screen.getByLabelText("Natural-language vehicle search").closest("form")!);

    await waitFor(() => expect(searchVehicles).toHaveBeenCalledWith({
      query: "How many vehicles are there?",
      run_id: "latest",
    }));
  });

  it("defaults to physical vehicles without loading experiment output", async () => {
    render(
      <MemoryRouter initialEntries={["/vehicles"]}>
        <VehicleSearchPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchVehicles).toHaveBeenCalled());

    expect(screen.getByRole("button", { name: "Physical Vehicles" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Raw Tracks" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Plate-Assisted Identity" })).toBeInTheDocument();
    expect(screen.getByText("VEHICLE_004")).toBeInTheDocument();
    expect(screen.getAllByText("HR38AD4296").length).toBeGreaterThan(0);
    expect(screen.getAllByText("TRACK_4").length).toBeGreaterThan(0);
    expect(screen.getByText("VEHICLE_004")).toHaveClass("table-badge", "table-badge--vehicle-id");
    expect(screen.getByText("TRACK_4")).toHaveClass("table-badge", "table-badge--track");
    expect(document.querySelector(".table-badge--class.badge-class--car")).not.toBeNull();
    expect(document.querySelector(".table-badge--colour.badge-tone--white")).not.toBeNull();
    expect(document.querySelector(".table-badge--status.badge-status--merged")).not.toBeNull();
    expect(fetchTrackReconciliation).not.toHaveBeenCalled();
    expect(fetchPlateAssistedVehicles).not.toHaveBeenCalled();
  });

  it("shows reconciled vehicle identities with original track fragments and counts", async () => {
    render(
      <MemoryRouter initialEntries={["/vehicles?run_id=20260808_182124"]}>
        <VehicleSearchPage />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Reconciled Vehicles" }));

    await waitFor(() => expect(fetchTrackReconciliation).toHaveBeenCalledWith("20260808_182124"));
    expect(screen.getByText("Raw ByteTrack Tracks")).toBeInTheDocument();
    expect(screen.getAllByText("Reconciled Vehicles").length).toBeGreaterThan(0);
    expect(screen.getByText("Recovered Fragments")).toBeInTheDocument();
    expect(screen.getAllByText("VEHICLE_037").length).toBeGreaterThan(0);
    expect(screen.getAllByText("TRACK_44").length).toBeGreaterThan(0);
    expect(screen.getAllByText("TRACK_48").length).toBeGreaterThan(0);
    expect(screen.getByText("YES")).toBeInTheDocument();
    expect(screen.getAllByText("0.748").length).toBeGreaterThan(0);
    expect(screen.getByText("View Evidence")).toBeInTheDocument();
    expect(document.querySelector(".table-badge--vehicle-id")).not.toBeNull();
    expect(document.querySelector(".table-badge--track")).not.toBeNull();
    expect(document.querySelector(".table-badge--status")).not.toBeNull();
  });

  it("shows a clean missing reconciliation message without breaking raw tracks", async () => {
    fetchTrackReconciliation.mockResolvedValue({
      run_id: "20260808_182124",
      available: false,
      message: "Reconciliation test has not been run for this run.",
      metrics: {},
      config: {},
      tracks: [],
      accepted_associations: [],
      manual_validation: [],
      visual_evidence: [],
      paths: {},
    });
    render(
      <MemoryRouter initialEntries={["/vehicles?run_id=20260808_182124"]}>
        <VehicleSearchPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(screen.getByText("VEHICLE_004")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Reconciled Vehicles" }));

    await waitFor(() => expect(screen.getByText("Reconciliation test has not been run for this run.")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Raw Tracks" }));
    expect(screen.getByText("TRACK_1")).toBeInTheDocument();
    expect(screen.getByText("DL8CAF5030")).toBeInTheDocument();
    expect(document.querySelector(".table-badge--status.badge-status--completed")).not.toBeNull();
  });

  it("loads plate-assisted identity as a separate experimental view", async () => {
    render(
      <MemoryRouter initialEntries={["/vehicles?run_id=20260808_182124"]}>
        <VehicleSearchPage />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Plate-Assisted Identity" }));

    await waitFor(() => expect(fetchPlateAssistedVehicles).toHaveBeenCalledWith("20260808_182124"));
    expect(screen.getByText("Plate-assisted identity experimental mode")).toBeInTheDocument();
    expect(screen.getByText("Plate-Assisted IDs")).toBeInTheDocument();
    expect(screen.getByText("Duplicates Removed")).toBeInTheDocument();
    expect(screen.getByText("Readable Plates")).toBeInTheDocument();
    expect(screen.getByText("Plate-Confirmed Merges")).toBeInTheDocument();
    expect(screen.getByText("VEHICLE_004")).toBeInTheDocument();
    expect(screen.getAllByText("HR38AD4296").length).toBeGreaterThan(0);
    expect(screen.getAllByText("TRACK_4").length).toBeGreaterThan(0);
    expect(screen.getAllByText("TRACK_14").length).toBeGreaterThan(0);
    expect(screen.getByText("PLATE_EXACT_MATCH | SPATIAL_MATCH")).toBeInTheDocument();
  });

  it("shows partial plate status without presenting it as an exact merge", async () => {
    render(
      <MemoryRouter initialEntries={["/vehicles?run_id=20260808_182124"]}>
        <VehicleSearchPage />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Plate-Assisted Identity" }));

    await waitFor(() => expect(screen.getByText("VEHICLE_010")).toBeInTheDocument());
    expect(screen.getAllByText("HR30T42").length).toBeGreaterThan(0);
    expect(screen.getAllByText("PARTIAL").length).toBeGreaterThan(0);
    expect(screen.getAllByText("TRACK_16").length).toBeGreaterThan(0);
  });

  it("shows a clean missing plate-assisted experiment state", async () => {
    fetchPlateAssistedVehicles.mockResolvedValue({
      run_id: "20260808_182124",
      experimental: true,
      stage: "plate_assisted_identity",
      available: false,
      message: "Plate-assisted identity experiment has not been run for this run.",
      verification: {},
      plate_coverage: {},
      baseline_without_plate: {},
      plate_assisted: {},
      vehicles: [],
      vehicle_id_map: {},
      track_plate_consensus: [],
      association_decisions: [],
      plate_pair_scores: [],
      identity_scores: [],
      paths: {},
    });
    render(
      <MemoryRouter initialEntries={["/vehicles?run_id=20260808_182124"]}>
        <VehicleSearchPage />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Plate-Assisted Identity" }));

    await waitFor(() => expect(screen.getByText("Plate-assisted identity experiment has not been run for this run.")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Raw Tracks" }));
    expect(screen.getByText("TRACK_1")).toBeInTheDocument();
  });
});
