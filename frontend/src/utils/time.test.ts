import { describe, expect, it } from "vitest";
import { formatVideoTime, parseVideoTime } from "./time";

describe("time utils", () => {
  it("formats seconds into operator-friendly timestamps", () => {
    expect(formatVideoTime(4.72)).toBe("00:04.72");
    expect(formatVideoTime(65.2)).toBe("01:05.20");
    expect(formatVideoTime(3665)).toBe("01:01:05");
  });

  it("parses video-relative time inputs", () => {
    expect(parseVideoTime("00:30")).toBe(30);
    expect(parseVideoTime("01:15")).toBe(75);
    expect(parseVideoTime("01:01:05")).toBe(3665);
    expect(parseVideoTime("")).toBeNull();
  });
});
