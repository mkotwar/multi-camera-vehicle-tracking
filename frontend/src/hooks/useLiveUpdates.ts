import { useEffect } from "react";

type LiveHandlers = {
  onCameraUpdate?: (payload: Record<string, unknown>) => void;
  onSystemStatus?: (payload: Record<string, unknown>) => void;
  onTrackColourUpdate?: (payload: Record<string, unknown>) => void;
  onSnapshot?: (payload: Record<string, unknown>) => void;
};

export function useLiveUpdates(handlers: LiveHandlers): void {
  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${protocol}://${window.location.host}/ws/live`);
    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data) as Record<string, unknown>;
      const type = String(payload.type ?? "");
      if (type === "camera_update" && handlers.onCameraUpdate) handlers.onCameraUpdate(payload);
      if (type === "system_status" && handlers.onSystemStatus) handlers.onSystemStatus(payload);
      if (type === "track_colour_update" && handlers.onTrackColourUpdate) handlers.onTrackColourUpdate(payload);
      if (type === "snapshot" && handlers.onSnapshot) handlers.onSnapshot(payload);
    };
    return () => socket.close();
  }, [handlers]);
}
