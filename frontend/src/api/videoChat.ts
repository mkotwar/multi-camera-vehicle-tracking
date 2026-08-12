import { apiPost } from "./client";
import type { VideoChatRequest, VideoChatResponse } from "../types/videoChat";

export function sendVideoChatMessage(request: VideoChatRequest): Promise<VideoChatResponse> {
  return apiPost<VideoChatResponse>("/api/video-chat", request);
}
