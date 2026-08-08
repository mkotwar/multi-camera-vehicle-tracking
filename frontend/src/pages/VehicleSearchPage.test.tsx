import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { VehicleSearchPage } from "./VehicleSearchPage";

const { fetchFilterOptions, fetchTracks } = vi.hoisted(() => ({
  fetchFilterOptions: vi.fn(),
  fetchTracks: vi.fn(),
}));

vi.mock("../api/filters", () => ({
  fetchFilterOptions,
}));

vi.mock("../api/tracks", () => ({
  fetchTracks,
}));

describe("VehicleSearchPage", () => {
  afterEach(() => {
    cleanup();
  });

  beforeEach(() => {
    fetchFilterOptions.mockReset();
    fetchTracks.mockReset();
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
        first_seen_seconds: 1.2,
        last_seen_seconds: 2.8,
        duration_seconds: 1.6,
        status: "COMPLETED",
      },
    ]);
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
});
