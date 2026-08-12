const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, detail: unknown) {
    super(`Request failed: ${status}`);
    this.status = status;
    this.detail = detail;
  }
}

export async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`);
  if (!response.ok) {
    throw await buildApiError(response);
  }
  return (await response.json()) as T;
}

export async function apiPost<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw await buildApiError(response);
  }
  return (await response.json()) as T;
}

export function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

async function buildApiError(response: Response): Promise<ApiError> {
  let detail: unknown = null;
  try {
    detail = await response.json();
  } catch {
    detail = await response.text();
  }
  return new ApiError(response.status, detail);
}
