import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { VideoChatPage } from "./VideoChatPage";

const { fetchRuns, sendVideoChatMessage } = vi.hoisted(() => ({
  fetchRuns: vi.fn(),
  sendVideoChatMessage: vi.fn(),
}));

vi.mock("../api/runs", () => ({
  fetchRuns,
}));

vi.mock("../api/videoChat", () => ({
  sendVideoChatMessage,
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

describe("VideoChatPage", () => {
  afterEach(() => {
    cleanup();
  });

  beforeEach(() => {
    fetchRuns.mockReset();
    sendVideoChatMessage.mockReset();
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

  it("renders evidence cards and View Track links for list queries", async () => {
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
      <MemoryRouter>
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
    expect(screen.getAllByRole("link", { name: "View Track" })[0]).toHaveAttribute("href", "/tracks/CAM_001/TRACK_13?run_id=20260812_113742");
    fireEvent.click(screen.getByRole("button", { name: "Show more" }));
    await waitFor(() => expect(sendVideoChatMessage).toHaveBeenLastCalledWith(expect.objectContaining({ message: "Show more" })));
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
});
