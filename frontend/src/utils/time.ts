export function formatVideoTime(seconds?: number | null): string {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) {
    return "Unavailable";
  }
  const value = Math.max(0, Number(seconds));
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const secs = value % 60;
  if (hours > 0) {
    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(Math.floor(secs)).padStart(2, "0")}`;
  }
  return `${String(minutes).padStart(2, "0")}:${secs.toFixed(2).padStart(5, "0")}`;
}

export function parseVideoTime(value: string): number | null {
  const input = String(value || "").trim();
  if (!input) {
    return null;
  }
  const parts = input.split(":");
  if (parts.some((part) => part.trim() === "")) {
    return null;
  }
  const numericParts = parts.map((part) => Number(part));
  if (numericParts.some((part) => Number.isNaN(part) || part < 0)) {
    return null;
  }
  if (numericParts.length === 1) {
    return numericParts[0];
  }
  if (numericParts.length === 2) {
    return numericParts[0] * 60 + numericParts[1];
  }
  if (numericParts.length === 3) {
    return numericParts[0] * 3600 + numericParts[1] * 60 + numericParts[2];
  }
  return null;
}

export function formatRelativeDate(value?: string | number | null): string {
  if (value === null || value === undefined || value === "") {
    return "Unavailable";
  }
  if (typeof value === "number") {
    return formatVideoTime(value);
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return date.toLocaleString();
}
