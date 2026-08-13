import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { VideoChatPage } from "./VideoChatPage";

const { fetchRuns, sendVideoChatMessage, fetchTrack, fetchTrackEvidence } = vi.hoisted(() => ({
  fetchRuns: vi.fn(),
  sendVideoChatMessage: vi.fn(),
  fetchTrack: vi.fn(),
  fetchTrackEvidence: vi.fn(),
}));

vi.mock("../api/runs", () => ({
  fetchRuns,
}));

vi.mock("../api/videoChat", () => ({
  sendVideoChatMessage,
}));

vi.mock("../api/tracks", () => ({
  fetchTrack,
  fetchTrackEvidence,
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

function LocationProbe() {
  const location = useLocation();
  return <span data-testid="location">{location.pathname}</span>;
}

describe("VideoChatPage", () => {
  afterEach(() => {
    cleanup();
  });

  beforeEach(() => {
    window.localStorage.clear();
    fetchRuns.mockReset();
    sendVideoChatMessage.mockReset();
    fetchTrack.mockReset();
    fetchTrackEvidence.mockReset();
    fetchRuns.mockResolvedValue([{ run_id: "20260812_113742", status: "COMPLETED", track_count: 41, camera_count: 1, duration_seconds: 20 }]);
    sendVideoChatMessage.mockResolvedValue({
      run_id: "20260812_113742",
      session_id: "session",
      answer: "There are 17 cars.",
      parser_used: "rule",
      parsed_query: {
        intent: "COUNT",
        include_classes: ["CAR"],
        exclude_classes: [],
        include_colours: [],
        exclude_colours: [],
        start_time: null,
        end_time: null,
        camera_id: null,
        group_by: null,
        comparison: null,
        show_evidence: false,
        context_reference: null,
      },
      analytics_result: { total: 17, vehicle_ids: ["CAM_001:TRACK_13"] },
      matching_vehicle_ids: ["CAM_001:TRACK_13"],
      evidence: [],
      context_used: false,
    });
    fetchTrack.mockImplementation((_cameraId: string, trackId: string, runId?: string) => Promise.resolve({
      run_id: runId ?? "20260812_113742",
      camera_id: "CAM_001",
      track_id: trackId,
      local_track_id: `CAM_001:${trackId}`,
      vehicle_class: trackId === "TRACK_13" ? "BUS" : "CAR",
      colour: trackId === "TRACK_13" ? "BLUE" : "WHITE",
      first_seen_seconds: trackId === "TRACK_13" ? 9.1 : 6.2,
      last_seen_seconds: trackId === "TRACK_13" ? 11.3 : 8.4,
      duration_seconds: 2.2,
      first_frame: trackId === "TRACK_13" ? 91 : 62,
      last_frame: trackId === "TRACK_13" ? 113 : 84,
      status: "COMPLETED",
      colour_resolution: [{ index: 1, label: trackId === "TRACK_13" ? "BLUE" : "WHITE", frame_number: 70, evidence_role: "SHARPEST" }],
    }));
    fetchTrackEvidence.mockResolvedValue([
      {
        frame_number: 70,
        timestamp_seconds: 7.0,
        evidence_role: "SHARPEST",
        crop_url: "/crop-sharp.jpg",
        full_frame_url: "/frame-sharp.jpg",
        colour_crop_result: "WHITE",
      },
      {
        frame_number: 84,
        timestamp_seconds: 8.4,
        evidence_role: "LAST",
        crop_url: "/crop-last.jpg",
        full_frame_url: "/frame-last.jpg",
        colour_crop_result: "WHITE",
      },
      {
        frame_number: 68,
        timestamp_seconds: 6.8,
        evidence_role: "HIGHEST_CONFIDENCE",
        crop_url: "/crop-conf.jpg",
        full_frame_url: "/frame-conf.jpg",
        colour_crop_result: "WHITE",
      },
    ]);
  });

  it("sends a user message and displays the assistant response", async () => {
    render(
      <MemoryRouter>
        <VideoChatPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchRuns).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText("Video chat message"), { target: { value: "How many cars were there?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(sendVideoChatMessage).toHaveBeenCalledTimes(1));
    expect(sendVideoChatMessage.mock.calls[0][0]).toMatchObject({
      message: "How many cars were there?",
      run_id: "20260812_113742",
    });
    const history = screen.getByLabelText("Video analytics chat history");
    expect(within(history).getByText("How many cars were there?")).toBeInTheDocument();
    expect(await within(history).findByText("There are 17 cars.")).toBeInTheDocument();
    expect(within(history).getByText("Analytics Assistant")).toBeInTheDocument();
  });

  it("renders a professional empty state with clickable prompt suggestions", async () => {
    render(
      <MemoryRouter>
        <VideoChatPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchRuns).toHaveBeenCalled());
    expect(screen.getByText("Ask questions about this video")).toBeInTheDocument();
    fireEvent.click(screen.getAllByRole("button", { name: "Show white cars" })[0]);

    await waitFor(() => expect(sendVideoChatMessage).toHaveBeenCalledWith(expect.objectContaining({ message: "Show white cars" })));
  });

  it("shows loading state while a message is in flight", async () => {
    let resolveChat: (value: unknown) => void = () => undefined;
    sendVideoChatMessage.mockReturnValue(new Promise((resolve) => { resolveChat = resolve; }));
    render(
      <MemoryRouter>
        <VideoChatPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchRuns).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText("Video chat message"), { target: { value: "Give me a summary" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(screen.getByRole("button", { name: "Sending..." })).toBeDisabled();
    expect(screen.getByText("Analytics Assistant is analyzing")).toBeInTheDocument();
    resolveChat({
      run_id: "20260812_113742",
      session_id: "session",
      answer: "Traffic summary: 41 completed unique vehicle records.",
      parser_used: "rule",
      parsed_query: { intent: "SUMMARY", include_classes: [], exclude_classes: [], include_colours: [], exclude_colours: [], show_evidence: false },
      analytics_result: { total_unique_vehicles: 41 },
      matching_vehicle_ids: [],
      evidence: [],
      context_used: false,
    });
    expect((await screen.findAllByText("Traffic summary: 41 completed unique vehicle records.")).length).toBeGreaterThan(0);
  });

  it("renders evidence cards and selects tracks inline without navigation", async () => {
    sendVideoChatMessage.mockResolvedValue({
      run_id: "20260812_113742",
      session_id: "session",
      answer: "6 white cars were observed.",
      parser_used: "rule",
      parsed_query: { intent: "LIST", include_classes: ["CAR"], exclude_classes: [], include_colours: ["WHITE"], exclude_colours: [], show_evidence: true },
      analytics_result: { total: 6 },
      matching_vehicle_ids: ["CAM_001:TRACK_13"],
      evidence_page: {
        matching_total: 2,
        evidence_returned_count: 1,
        evidence_offset: 0,
        evidence_page_size: 1,
        evidence_remaining_count: 1,
        shown_count: 1,
        next_offset: 1,
      },
      evidence: [
        {
          vehicle_id: "CAM_001:TRACK_13",
          camera_id: "CAM_001",
          track_id: "TRACK_13",
          vehicle_class: "CAR",
          colour: "WHITE",
          first_seen_seconds: 6.2,
          last_seen_seconds: 8.4,
          image_url: "/api/media/florence_selected_crops/20260812_113742/CAM_001/TRACK_13/frame.jpg",
          track_detail_url: "/tracks/CAM_001/TRACK_13?run_id=20260812_113742",
        },
      ],
      context_used: false,
    });
    render(
      <MemoryRouter initialEntries={["/video-chat"]}>
        <LocationProbe />
        <VideoChatPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchRuns).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText("Video chat message"), { target: { value: "Show me the white cars" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect((await screen.findAllByText("TRACK_13")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("CAR").length).toBeGreaterThan(0);
    expect(screen.getAllByText("WHITE").length).toBeGreaterThan(0);
    expect(screen.getByText("Showing 1 of 2 vehicles")).toBeInTheDocument();
    fireEvent.click(screen.getAllByRole("button", { name: "View Track" })[0]);
    await waitFor(() => expect(fetchTrack).toHaveBeenCalledWith("CAM_001", "TRACK_13", "20260812_113742"));
    expect(screen.getByTestId("location")).toHaveTextContent("/video-chat");
    expect(await screen.findByText("CAM_001:TRACK_13")).toBeInTheDocument();
    expect(screen.getByText("Open full track page")).toHaveAttribute("href", "/tracks/CAM_001/TRACK_13?run_id=20260812_113742");
    fireEvent.click(screen.getByRole("button", { name: "Show more" }));
    await waitFor(() => expect(sendVideoChatMessage).toHaveBeenLastCalledWith(expect.objectContaining({ message: "Show more" })));
  });

  it("switches selected tracks and preserves chat results", async () => {
    sendVideoChatMessage.mockResolvedValue({
      run_id: "20260812_113742",
      session_id: "session",
      answer: "2 vehicles were observed.",
      parser_used: "rule",
      parsed_query: { intent: "LIST", include_classes: [], exclude_classes: [], include_colours: [], exclude_colours: [], show_evidence: true },
      analytics_result: { total: 2 },
      matching_vehicle_ids: ["CAM_001:TRACK_5", "CAM_001:TRACK_13"],
      evidence: [
        {
          vehicle_id: "CAM_001:TRACK_5",
          camera_id: "CAM_001",
          track_id: "TRACK_5",
          vehicle_class: "CAR",
          colour: "WHITE",
          first_seen_seconds: 1,
          last_seen_seconds: 2,
          image_url: "/track-5.jpg",
          track_detail_url: "/tracks/CAM_001/TRACK_5?run_id=20260812_113742",
        },
        {
          vehicle_id: "CAM_001:TRACK_13",
          camera_id: "CAM_001",
          track_id: "TRACK_13",
          vehicle_class: "BUS",
          colour: "BLUE",
          first_seen_seconds: 3,
          last_seen_seconds: 5,
          image_url: "/track-13.jpg",
          track_detail_url: "/tracks/CAM_001/TRACK_13?run_id=20260812_113742",
        },
      ],
      context_used: false,
    });
    render(
      <MemoryRouter initialEntries={["/video-chat"]}>
        <VideoChatPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchRuns).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText("Video chat message"), { target: { value: "Show vehicles" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Show vehicles")).toBeInTheDocument();
    expect((await screen.findAllByText("2 vehicles were observed.")).length).toBeGreaterThan(0);
    fireEvent.click(screen.getAllByRole("button", { name: "View Track" })[0]);
    expect(await screen.findByText("CAM_001:TRACK_5")).toBeInTheDocument();
    fireEvent.click(screen.getAllByRole("button", { name: "View Track" })[0]);
    expect(await screen.findByText("CAM_001:TRACK_13")).toBeInTheDocument();
    expect(screen.getByText("Show vehicles")).toBeInTheDocument();
    expect(screen.getAllByText("2 vehicles were observed.").length).toBeGreaterThan(0);
    expect(screen.getAllByText("TRACK_5").length).toBeGreaterThan(0);
    expect(screen.getAllByText("TRACK_13").length).toBeGreaterThan(0);
  });

  it("renders selected track evidence categories", async () => {
    sendVideoChatMessage.mockResolvedValue({
      run_id: "20260812_113742",
      session_id: "session",
      answer: "1 vehicle was observed.",
      parser_used: "rule",
      parsed_query: { intent: "LIST", include_classes: [], exclude_classes: [], include_colours: [], exclude_colours: [], show_evidence: true },
      analytics_result: { total: 1 },
      matching_vehicle_ids: ["CAM_001:TRACK_5"],
      evidence: [
        {
          vehicle_id: "CAM_001:TRACK_5",
          camera_id: "CAM_001",
          track_id: "TRACK_5",
          vehicle_class: "CAR",
          colour: "WHITE",
          image_url: "/track-5.jpg",
          track_detail_url: "/tracks/CAM_001/TRACK_5?run_id=20260812_113742",
        },
      ],
      context_used: false,
    });
    render(
      <MemoryRouter initialEntries={["/video-chat"]}>
        <VideoChatPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchRuns).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText("Video chat message"), { target: { value: "Show white cars" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    fireEvent.click(await screen.findByRole("button", { name: "View Track" }));

    expect(await screen.findByText("SHARPEST")).toBeInTheDocument();
    expect(screen.getByText("LAST")).toBeInTheDocument();
    expect(screen.getByText("HIGHEST CONFIDENCE")).toBeInTheDocument();
    expect(screen.getAllByText("Full Frame").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Vehicle Crop").length).toBeGreaterThan(0);
  });

  it("shows inline track loading and error states", async () => {
    let rejectTrack: (reason?: unknown) => void = () => undefined;
    fetchTrack.mockReturnValue(new Promise((_resolve, reject) => { rejectTrack = reject; }));
    sendVideoChatMessage.mockResolvedValue({
      run_id: "20260812_113742",
      session_id: "session",
      answer: "1 vehicle was observed.",
      parser_used: "rule",
      parsed_query: { intent: "LIST", include_classes: [], exclude_classes: [], include_colours: [], exclude_colours: [], show_evidence: true },
      analytics_result: { total: 1 },
      matching_vehicle_ids: ["CAM_001:TRACK_5"],
      evidence: [
        {
          vehicle_id: "CAM_001:TRACK_5",
          camera_id: "CAM_001",
          track_id: "TRACK_5",
          vehicle_class: "CAR",
          colour: "WHITE",
          image_url: "/track-5.jpg",
          track_detail_url: "/tracks/CAM_001/TRACK_5?run_id=20260812_113742",
        },
      ],
      context_used: false,
    });
    render(
      <MemoryRouter initialEntries={["/video-chat"]}>
        <VideoChatPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchRuns).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText("Video chat message"), { target: { value: "Show white cars" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    fireEvent.click(await screen.findByRole("button", { name: "View Track" }));

    expect(screen.getByText("Loading track details...")).toBeInTheDocument();
    rejectTrack(new Error("failed"));
    expect(await screen.findByText("Unable to load track details.")).toBeInTheDocument();
    expect(screen.getByText("1 vehicle was observed.")).toBeInTheDocument();
  });

  it("offers evidence follow-up for count responses without automatic cards", async () => {
    render(
      <MemoryRouter>
        <VideoChatPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchRuns).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText("Video chat message"), { target: { value: "How many cars were there?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect((await screen.findAllByText("There are 17 cars.")).length).toBeGreaterThan(0);
    expect(screen.queryByText("TRACK_13")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Show evidence" }));
    await waitFor(() => expect(sendVideoChatMessage).toHaveBeenLastCalledWith(expect.objectContaining({ message: "Show them" })));
  });

  it("does not offer evidence actions for general chat responses", async () => {
    sendVideoChatMessage.mockResolvedValue({
      run_id: "20260812_113742",
      session_id: "session",
      original_query: "hello",
      answer: "Hello. I can answer questions about this processed traffic video.",
      parser_used: "rule_based",
      parsed_query: {
        intent: "GENERAL_CHAT",
        include_classes: [],
        exclude_classes: [],
        include_colours: [],
        exclude_colours: [],
        show_evidence: false,
        context_reference: null,
      },
      analytics_result: {},
      matching_vehicle_ids: [],
      evidence: [],
      context_used: false,
    });
    render(
      <MemoryRouter>
        <VideoChatPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchRuns).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText("Video chat message"), { target: { value: "hello" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Hello. I can answer questions about this processed traffic video.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Show evidence" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Show more" })).not.toBeInTheDocument();
  });

  it("renders summary metrics from structured analytics", async () => {
    sendVideoChatMessage.mockResolvedValue({
      run_id: "20260812_113742",
      session_id: "session",
      answer: "Traffic summary: 41 completed unique vehicle records.",
      parser_used: "rule",
      parsed_query: { intent: "SUMMARY", include_classes: [], exclude_classes: [], include_colours: [], exclude_colours: [], show_evidence: false },
      analytics_result: { total_unique_vehicles: 41, by_type: { MOTORCYCLE: 18, CAR: 17 } },
      matching_vehicle_ids: [],
      evidence: [],
      context_used: false,
    });
    render(
      <MemoryRouter>
        <VideoChatPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchRuns).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText("Video chat message"), { target: { value: "Give me a traffic summary" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Total vehicles")).toBeInTheDocument();
    expect(screen.getByText("MOTORCYCLE")).toBeInTheDocument();
    expect(screen.getByText("CAR")).toBeInTheDocument();
  });

  it("renders comparison metrics and keeps developer details collapsed", async () => {
    sendVideoChatMessage.mockResolvedValue({
      run_id: "20260812_113742",
      session_id: "session",
      original_query: "Were bikes more common than cars?",
      answer: "Motorcycles were slightly more common.",
      parser_used: "rule",
      parsed_query: { intent: "COMPARE", include_classes: [], exclude_classes: [], include_colours: [], exclude_colours: [], show_evidence: false },
      analytics_result: { difference: 1, counts: { MOTORCYCLE: 18, CAR: 17 } },
      matching_vehicle_ids: [],
      evidence: [],
      context_used: false,
    });
    render(
      <MemoryRouter>
        <VideoChatPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchRuns).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText("Video chat message"), { target: { value: "Were bikes more common than cars?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Difference")).toBeInTheDocument();
    const details = screen.getByText("Developer Details").closest("details");
    expect(details).not.toHaveAttribute("open");
  });

  it("keeps conversation history and renders errors", async () => {
    sendVideoChatMessage.mockRejectedValue(new Error("I could not understand that question."));
    render(
      <MemoryRouter>
        <VideoChatPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchRuns).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText("Video chat message"), { target: { value: "show dark vehicles" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("show dark vehicles")).toBeInTheDocument();
    expect(await screen.findByText("I could not understand that question.")).toBeInTheDocument();
  });

  it("restores a persisted conversation after remount", async () => {
    const { unmount } = render(
      <MemoryRouter>
        <VideoChatPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchRuns).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText("Video chat message"), { target: { value: "How many cars were there?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect((await screen.findAllByText("There are 17 cars.")).length).toBeGreaterThan(0);

    unmount();
    render(
      <MemoryRouter>
        <VideoChatPage />
      </MemoryRouter>,
    );

    const history = screen.getByLabelText("Video analytics chat history");
    expect(within(history).getByText("How many cars were there?")).toBeInTheDocument();
    expect(within(history).getAllByText("There are 17 cars.").length).toBeGreaterThan(0);
  });

  it("continues a restored conversation with the same session id", async () => {
    const { unmount } = render(
      <MemoryRouter>
        <VideoChatPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchRuns).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText("Video chat message"), { target: { value: "Show white cars" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(sendVideoChatMessage).toHaveBeenCalledTimes(1));
    const restoredSessionId = sendVideoChatMessage.mock.calls[0][0].session_id;

    unmount();
    render(
      <MemoryRouter>
        <VideoChatPage />
      </MemoryRouter>,
    );

    fireEvent.change(screen.getByLabelText("Video chat message"), { target: { value: "How many of those lasted more than 2 seconds?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(sendVideoChatMessage).toHaveBeenCalledTimes(2));
    expect(sendVideoChatMessage.mock.calls[1][0]).toMatchObject({
      message: "How many of those lasted more than 2 seconds?",
      run_id: "20260812_113742",
      session_id: restoredSessionId,
    });
  });

  it("starts a new active session when New Chat is clicked", async () => {
    render(
      <MemoryRouter>
        <VideoChatPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchRuns).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText("Video chat message"), { target: { value: "How many cars were there?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(sendVideoChatMessage).toHaveBeenCalledTimes(1));
    const firstSessionId = sendVideoChatMessage.mock.calls[0][0].session_id;
    expect((await screen.findAllByText("There are 17 cars.")).length).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole("button", { name: "New Chat" }));
    expect(screen.getByLabelText("Video analytics chat history").querySelectorAll(".chat-message-row")).toHaveLength(0);
    expect(screen.getByText("Ask questions about this video")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Video chat message"), { target: { value: "Show them" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(sendVideoChatMessage).toHaveBeenCalledTimes(2));
    expect(sendVideoChatMessage.mock.calls[1][0].session_id).not.toEqual(firstSessionId);
  });

  it("loads safely with malformed persisted state", async () => {
    window.localStorage.setItem("video-chat.active-run-id.v1", "broken-run");
    window.localStorage.setItem("video-chat.sessions-by-run.v1", "{not-json");

    render(
      <MemoryRouter>
        <VideoChatPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchRuns).toHaveBeenCalled());
    expect(screen.getByText("Ask questions about this video")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Video chat message"), { target: { value: "How many cars were there?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(sendVideoChatMessage).toHaveBeenCalledTimes(1));
  });

  it("restores the persisted run instead of switching follow-ups to a different latest run", async () => {
    fetchRuns.mockResolvedValue([
      { run_id: "RUN_B", status: "COMPLETED", track_count: 3, camera_count: 1, duration_seconds: 4 },
      { run_id: "RUN_A", status: "COMPLETED", track_count: 8, camera_count: 1, duration_seconds: 10 },
    ]);
    window.localStorage.setItem("video-chat.active-run-id.v1", "RUN_A");
    window.localStorage.setItem("video-chat.sessions-by-run.v1", JSON.stringify({
      RUN_A: {
        version: 1,
        session_id: "persisted-run-a",
        run_id: "RUN_A",
        created_at: "2026-08-13T00:00:00.000Z",
        updated_at: "2026-08-13T00:00:00.000Z",
        messages: [
          { id: "user-a", role: "user", text: "show white cars", timestamp: "2026-08-13T00:00:00.000Z" },
          { id: "assistant-a", role: "assistant", text: "6 white cars found.", timestamp: "2026-08-13T00:00:01.000Z", evidence: [], debug: {
            original_query: "show white cars",
            parser_used: "rule",
            parsed_query: { intent: "LIST", include_classes: ["CAR"], exclude_classes: [], include_colours: ["WHITE"], exclude_colours: [], show_evidence: false },
            analytics_result: { total: 6 },
            matching_vehicle_ids: ["CAM_001:TRACK_1"],
            context_used: false,
          } },
        ],
      },
    }));

    render(
      <MemoryRouter>
        <VideoChatPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchRuns).toHaveBeenCalled());
    expect(screen.getByLabelText("Video chat run")).toHaveValue("RUN_A");
    fireEvent.change(screen.getByLabelText("Video chat message"), { target: { value: "how many of those are visible for more than 2 seconds?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(sendVideoChatMessage).toHaveBeenCalledTimes(1));
    expect(sendVideoChatMessage.mock.calls[0][0]).toMatchObject({ run_id: "RUN_A", session_id: "persisted-run-a" });
  });

  it("keeps independent histories when switching between runs", async () => {
    fetchRuns.mockResolvedValue([
      { run_id: "RUN_A", status: "COMPLETED", track_count: 8, camera_count: 1, duration_seconds: 10 },
      { run_id: "RUN_B", status: "COMPLETED", track_count: 3, camera_count: 1, duration_seconds: 4 },
    ]);
    sendVideoChatMessage
      .mockResolvedValueOnce({
        run_id: "RUN_A",
        session_id: "session-a",
        answer: "Run A had 8 cars.",
        parser_used: "rule",
        parsed_query: { intent: "COUNT", include_classes: ["CAR"], exclude_classes: [], include_colours: [], exclude_colours: [], show_evidence: false },
        analytics_result: { total: 8 },
        matching_vehicle_ids: [],
        evidence: [],
        context_used: false,
      })
      .mockResolvedValueOnce({
        run_id: "RUN_B",
        session_id: "session-b",
        answer: "Run B had 3 motorcycles.",
        parser_used: "rule",
        parsed_query: { intent: "COUNT", include_classes: ["MOTORCYCLE"], exclude_classes: [], include_colours: [], exclude_colours: [], show_evidence: false },
        analytics_result: { total: 3 },
        matching_vehicle_ids: [],
        evidence: [],
        context_used: false,
      });

    render(
      <MemoryRouter>
        <VideoChatPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(screen.getByLabelText("Video chat run")).toHaveValue("RUN_A"));
    fireEvent.change(screen.getByLabelText("Video chat message"), { target: { value: "How many cars were there?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect((await screen.findAllByText("Run A had 8 cars.")).length).toBeGreaterThan(0);

    fireEvent.change(screen.getByLabelText("Video chat run"), { target: { value: "RUN_B" } });
    expect(screen.getByLabelText("Video analytics chat history").querySelectorAll(".chat-message-row")).toHaveLength(0);
    fireEvent.change(screen.getByLabelText("Video chat message"), { target: { value: "Show motorcycles." } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect((await screen.findAllByText("Run B had 3 motorcycles.")).length).toBeGreaterThan(0);

    fireEvent.change(screen.getByLabelText("Video chat run"), { target: { value: "RUN_A" } });
    let history = screen.getByLabelText("Video analytics chat history");
    expect(within(history).getByText("How many cars were there?")).toBeInTheDocument();
    expect(within(history).getAllByText("Run A had 8 cars.").length).toBeGreaterThan(0);
    expect(within(history).queryByText("Show motorcycles.")).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Video chat run"), { target: { value: "RUN_B" } });
    history = screen.getByLabelText("Video analytics chat history");
    expect(within(history).getByText("Show motorcycles.")).toBeInTheDocument();
    expect(within(history).getAllByText("Run B had 3 motorcycles.").length).toBeGreaterThan(0);
    expect(within(history).queryByText("How many cars were there?")).not.toBeInTheDocument();
  });

  it("does not overwrite existing run history with an initial empty state", async () => {
    fetchRuns.mockResolvedValue([{ run_id: "RUN_A", status: "COMPLETED", track_count: 8, camera_count: 1, duration_seconds: 10 }]);
    const storedSession = {
      version: 1,
      session_id: "stored-run-a",
      run_id: "RUN_A",
      created_at: "2026-08-13T00:00:00.000Z",
      updated_at: "2026-08-13T00:00:00.000Z",
      messages: [
        { id: "user-a", role: "user", text: "First stored question", timestamp: "2026-08-13T00:00:00.000Z" },
        { id: "assistant-a", role: "assistant", text: "First stored answer", timestamp: "2026-08-13T00:00:01.000Z", evidence: [], debug: {
          original_query: "First stored question",
          parser_used: "rule",
          parsed_query: { intent: "COUNT", include_classes: [], exclude_classes: [], include_colours: [], exclude_colours: [], show_evidence: false },
          analytics_result: { total: 1 },
          matching_vehicle_ids: [],
          context_used: false,
        } },
      ],
    };
    window.localStorage.setItem("video-chat.sessions-by-run.v1", JSON.stringify({ RUN_A: storedSession }));

    render(
      <MemoryRouter>
        <VideoChatPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(screen.getByLabelText("Video chat run")).toHaveValue("RUN_A"));
    const history = screen.getByLabelText("Video analytics chat history");
    expect(within(history).getByText("First stored question")).toBeInTheDocument();
    expect(within(history).getAllByText("First stored answer").length).toBeGreaterThan(0);
    const persisted = JSON.parse(window.localStorage.getItem("video-chat.sessions-by-run.v1") || "{}");
    expect(persisted.RUN_A.messages).toHaveLength(2);
  });

  it("does not duplicate messages after remount", async () => {
    const { unmount } = render(
      <MemoryRouter>
        <VideoChatPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchRuns).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText("Video chat message"), { target: { value: "How many cars were there?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(sendVideoChatMessage).toHaveBeenCalledTimes(1));

    unmount();
    render(
      <MemoryRouter>
        <VideoChatPage />
      </MemoryRouter>,
    );

    const rows = screen.getByLabelText("Video analytics chat history").querySelectorAll(".chat-message-row");
    expect(rows).toHaveLength(2);
  });

  it("track selection does not rewrite the persisted conversation", async () => {
    sendVideoChatMessage.mockResolvedValue({
      run_id: "20260812_113742",
      session_id: "session",
      answer: "1 vehicle was observed.",
      parser_used: "rule",
      parsed_query: { intent: "LIST", include_classes: [], exclude_classes: [], include_colours: [], exclude_colours: [], show_evidence: true },
      analytics_result: { total: 1 },
      matching_vehicle_ids: ["CAM_001:TRACK_5"],
      evidence: [
        {
          vehicle_id: "CAM_001:TRACK_5",
          camera_id: "CAM_001",
          track_id: "TRACK_5",
          vehicle_class: "CAR",
          colour: "WHITE",
          image_url: "/track-5.jpg",
          track_detail_url: "/tracks/CAM_001/TRACK_5?run_id=20260812_113742",
        },
      ],
      context_used: false,
    });
    render(
      <MemoryRouter>
        <VideoChatPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchRuns).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText("Video chat message"), { target: { value: "Show vehicles" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await screen.findByText("TRACK_5");
    const beforeSelection = window.localStorage.getItem("video-chat.sessions-by-run.v1");

    fireEvent.click(screen.getByRole("button", { name: "View Track" }));
    expect(await screen.findByText("CAM_001:TRACK_5")).toBeInTheDocument();
    expect(window.localStorage.getItem("video-chat.sessions-by-run.v1")).toEqual(beforeSelection);
  });

  it("restores messages with evidence cards after remount", async () => {
    sendVideoChatMessage.mockResolvedValue({
      run_id: "20260812_113742",
      session_id: "session",
      answer: "1 white car was observed.",
      parser_used: "rule",
      parsed_query: { intent: "LIST", include_classes: ["CAR"], exclude_classes: [], include_colours: ["WHITE"], exclude_colours: [], show_evidence: true },
      analytics_result: { total: 1 },
      matching_vehicle_ids: ["CAM_001:TRACK_5"],
      evidence_page: {
        matching_total: 1,
        evidence_returned_count: 1,
        evidence_offset: 0,
        evidence_page_size: 6,
        evidence_remaining_count: 0,
        shown_count: 1,
        next_offset: 1,
      },
      evidence: [
        {
          vehicle_id: "CAM_001:TRACK_5",
          camera_id: "CAM_001",
          track_id: "TRACK_5",
          vehicle_class: "CAR",
          colour: "WHITE",
          first_seen_seconds: 1,
          last_seen_seconds: 3,
          image_url: "/track-5.jpg",
          track_detail_url: "/tracks/CAM_001/TRACK_5?run_id=20260812_113742",
        },
      ],
      context_used: false,
    });
    const { unmount } = render(
      <MemoryRouter>
        <VideoChatPage />
      </MemoryRouter>,
    );

    await waitFor(() => expect(fetchRuns).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText("Video chat message"), { target: { value: "Show white cars" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("TRACK_5")).toBeInTheDocument();

    unmount();
    render(
      <MemoryRouter>
        <VideoChatPage />
      </MemoryRouter>,
    );

    expect(screen.getAllByText("1 white car was observed.").length).toBeGreaterThan(0);
    expect(screen.getByText("TRACK_5")).toBeInTheDocument();
    expect(screen.getAllByText("WHITE").length).toBeGreaterThan(0);
    expect(screen.getByText("All 1 matching vehicles shown.")).toBeInTheDocument();
  });
});
