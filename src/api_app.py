from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .ollama_qwen_provider import build_chat_llm_provider_from_env
from .run_repository import RunRepository
from .runtime_state import get_runtime_state_manager
from .video_chat import handle_video_chat
from .vehicle_nlp import VehicleQueryParseError, search_vehicle_data


class VehicleSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    run_id: str | None = "latest"


class VideoChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    run_id: str | None = "latest"
    session_id: str | None = None


def create_app(*, outputs_root: str | Path = "outputs/runs") -> Any:
    try:
        from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse, JSONResponse, Response
        from fastapi.staticfiles import StaticFiles
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("FastAPI is not installed. Install fastapi and uvicorn first.") from exc

    repository = RunRepository(outputs_root)
    runtime_state = get_runtime_state_manager()
    chat_sessions: dict[str, dict[str, Any]] = {}
    chat_llm_provider = build_chat_llm_provider_from_env()
    app = FastAPI(title="Multi-Camera Vehicle Tracking API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    frontend_dist = Path("frontend/dist")
    if frontend_dist.exists():
        app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    def _build_media_url(media: dict[str, Any] | None) -> str | None:
        if not media:
            return None
        category = str(media.get("category") or "").strip()
        run_id = str(media.get("run_id") or "").strip()
        parts = [str(item).strip() for item in list(media.get("parts", []) or []) if str(item).strip()]
        if not category or not run_id or not parts:
            return None
        return f"/api/media/{category}/{run_id}/{'/'.join(parts)}"

    def _serialize_track(track: dict[str, Any]) -> dict[str, Any]:
        payload = dict(track)
        payload["best_crop_url"] = _build_media_url(payload.get("best_crop_parts"))
        payload["evidence"] = [_serialize_evidence_item(item) for item in list(payload.get("evidence", []) or [])]
        payload["colour_resolution"] = list(payload.get("colour_resolution", []) or [])
        return payload

    def _serialize_evidence_item(item: dict[str, Any]) -> dict[str, Any]:
        payload = dict(item)
        payload["crop_url"] = _build_media_url(payload.get("crop_media"))
        payload["full_frame_url"] = _build_media_url(payload.get("full_frame_media"))
        return payload

    def _persisted_system_status() -> dict[str, Any]:
        latest_run_id = repository.latest_run_id()
        if latest_run_id is None:
            return runtime_state.get_system_status()
        run = repository.get_run(latest_run_id) or {}
        summary = dict(run.get("summary", {}) or {})
        detection = dict(run.get("detection_tracking_metrics", {}) or {})
        enrichment = dict(run.get("vehicle_enrichment_metrics", {}) or {})
        cameras = repository.list_cameras(run_id=latest_run_id)
        return {
            "pipeline_status": str(summary.get("status") or "completed").lower(),
            "camera_count": len(cameras),
            "processing_camera_count": len(cameras),
            "online_camera_count": len(cameras),
            "processed_fps": float(detection.get("detection_frames_total", 0) or 0) / max(float(detection.get("duration_seconds", 0.0) or 0.0), 1.0),
            "yolo_status": "healthy" if summary else "unavailable",
            "colour_worker_status": "healthy" if summary.get("vehicle_enrichment_enabled") else "unavailable",
            "colour_queue_depth": int(summary.get("pending_colour_jobs_at_shutdown", 0) or 0),
            "colour_queue_capacity": int(summary.get("colour_queue_size", 0) or 0),
            "pending_colour_jobs": int(summary.get("pending_colour_jobs_at_shutdown", 0) or 0),
            "cache_misses": int(summary.get("cache_misses", 0) or 0),
            "frame_loss": int(summary.get("frame_loss", 0) or 0),
            "order_violations": int(detection.get("frame_order_violations", 0) or 0),
            "last_update": str(run.get("metadata", {}).get("completed_at") or run.get("metadata", {}).get("started_at") or ""),
            "run_id": latest_run_id,
            "track_count": int(run.get("track_count", 0) or 0),
            "colour_queue_peak_depth": int(enrichment.get("colour_queue_peak_depth", 0) or 0),
            "average_colour_calls_per_track": enrichment.get("average_colour_calls_per_track"),
            "yolo_image_size": detection.get("image_size"),
            "yolo_batch_size": detection.get("detection_batch_size_configured"),
        }

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        current_run = runtime_state.get_current_run()
        return {
            "status": "ok",
            "pipeline_status": current_run.get("pipeline_status"),
            "run_id": current_run.get("run_id"),
            "run_directory": current_run.get("run_directory"),
        }

    @app.get("/api/cameras")
    def list_cameras(run_id: str | None = None) -> list[dict[str, Any]]:
        cameras = runtime_state.list_cameras()
        if cameras:
            return cameras
        persisted = repository.list_cameras(run_id=run_id or "latest")
        for item in persisted:
            parts = list(item.get("latest_frame_url_parts") or [])
            if parts:
                item["frame_url"] = f"/api/media/tracked_frames/{item['run_id']}/{'/'.join(parts)}"
        return persisted

    @app.get("/api/cameras/{camera_id}")
    def get_camera(camera_id: str, run_id: str | None = None) -> dict[str, Any]:
        camera = runtime_state.get_camera(camera_id)
        if camera is not None:
            return camera
        for item in repository.list_cameras(run_id=run_id or "latest"):
            if str(item.get("camera_id")) == camera_id:
                return item
        raise HTTPException(status_code=404, detail="Camera not found")

    @app.get("/api/cameras/{camera_id}/frame")
    def get_camera_frame(camera_id: str, run_id: str | None = None) -> Response:
        frame_bytes = runtime_state.get_frame_bytes(camera_id)
        if frame_bytes is not None:
            return Response(content=frame_bytes, media_type="image/jpeg")
        for item in repository.list_cameras(run_id=run_id or "latest"):
            if str(item.get("camera_id")) != camera_id:
                continue
            parts = list(item.get("latest_frame_url_parts") or [])
            if not parts:
                break
            target = repository.resolve_media_path(run_id=str(item.get("run_id")), category="tracked_frames", relative_parts=parts)
            if target is not None:
                return FileResponse(str(target))
        raise HTTPException(status_code=404, detail="Frame not available")

    @app.get("/api/tracks")
    def list_tracks(
        run_id: str | None = None,
        camera_id: str | None = None,
        vehicle_class: str | None = None,
        colour: str | None = None,
        track_id: str | None = None,
        from_time: float | None = Query(default=None),
        to_time: float | None = Query(default=None),
    ) -> list[dict[str, Any]]:
        persisted = repository.list_tracks(
            run_id=run_id,
            camera_id=camera_id,
            vehicle_class=vehicle_class,
            colour=colour,
            track_id=track_id,
            from_time=from_time,
            to_time=to_time,
        )
        runtime_tracks = runtime_state.list_tracks()
        runtime_payload: list[dict[str, Any]] = []
        for item in runtime_tracks:
            payload = {
                "run_id": runtime_state.get_current_run().get("run_id"),
                "camera_id": item.get("camera_id"),
                "track_id": item.get("track_id"),
                "local_track_id": item.get("local_track_id"),
                "status": item.get("status"),
                "vehicle_class": item.get("vehicle_class"),
                "colour": item.get("colour"),
                "colour_status": item.get("colour_status"),
                "first_seen": item.get("first_seen"),
                "last_seen": item.get("last_seen"),
                "first_frame": None,
                "last_frame": item.get("frame_number"),
                "observation_count": None,
                "completion_reason": None,
                "vehicle_enrichment_status": item.get("colour_status"),
                "evidence": item.get("evidence", []),
                "best_crop": None,
                "available_crop_paths": [],
                "first_seen_seconds": item.get("first_seen"),
                "last_seen_seconds": item.get("last_seen"),
                "duration_seconds": (
                    max(0.0, float(item.get("last_seen")) - float(item.get("first_seen")))
                    if item.get("first_seen") is not None and item.get("last_seen") is not None
                    else None
                ),
                "runtime": True,
            }
            runtime_payload.append(payload)
        merged = {str(item.get("local_track_id")): item for item in persisted}
        for item in runtime_payload:
            merged[str(item.get("local_track_id"))] = item
        rows = list(merged.values())
        if camera_id:
            rows = [item for item in rows if str(item.get("camera_id")) == camera_id]
        if vehicle_class:
            rows = [item for item in rows if str(item.get("vehicle_class", "")).upper() == vehicle_class.upper()]
        if colour:
            rows = [item for item in rows if str(item.get("colour", "")).upper() == colour.upper()]
        if track_id:
            rows = [item for item in rows if str(item.get("track_id")) == track_id or str(item.get("local_track_id")) == track_id]
        rows.sort(key=lambda item: (str(item.get("run_id", "")), float(item.get("last_seen_seconds") or item.get("last_seen") or 0.0)), reverse=True)
        return [_serialize_track(item) for item in rows]

    @app.get("/api/tracks/{camera_id}/{track_id}")
    def get_track(camera_id: str, track_id: str, run_id: str | None = None) -> dict[str, Any]:
        runtime_tracks = runtime_state.list_tracks()
        for item in runtime_tracks:
            if str(item.get("camera_id")) == camera_id and (
                str(item.get("track_id")) == track_id or str(item.get("local_track_id")) == track_id
            ):
                return _serialize_track(item)
        track = repository.get_track(camera_id=camera_id, track_id=track_id, run_id=run_id)
        if track is None:
            raise HTTPException(status_code=404, detail="Track not found")
        return _serialize_track(track)

    @app.get("/api/tracks/{camera_id}/{track_id}/evidence")
    def get_track_evidence(camera_id: str, track_id: str, run_id: str | None = None) -> list[dict[str, Any]]:
        runtime_track = None
        for item in runtime_state.list_tracks():
            if str(item.get("camera_id")) == camera_id and (
                str(item.get("track_id")) == track_id or str(item.get("local_track_id")) == track_id
            ):
                runtime_track = item
                break
        if runtime_track is not None:
            return [_serialize_evidence_item(item) for item in list(runtime_track.get("evidence", []) or [])]
        evidence = repository.get_track_evidence(camera_id=camera_id, track_id=track_id, run_id=run_id)
        if not evidence:
            raise HTTPException(status_code=404, detail="Track evidence not found")
        return [_serialize_evidence_item(item) for item in evidence]

    @app.post("/api/vehicle-search")
    def vehicle_search(request: VehicleSearchRequest) -> dict[str, Any]:
        resolved_run_id = repository.resolve_run_id(request.run_id)
        if resolved_run_id is None:
            raise HTTPException(status_code=404, detail={"error": "run_not_found", "detail": "Run not found"})
        tracks_path = repository.tracks_json_path(resolved_run_id)
        if tracks_path is None or not tracks_path.exists():
            raise HTTPException(
                status_code=500,
                detail={"error": "tracks_json_missing", "detail": f"tracks.json not found for run {resolved_run_id}"},
            )
        try:
            payload = search_vehicle_data(query=request.query, tracks_path=tracks_path)
        except VehicleQueryParseError as exc:
            raise HTTPException(status_code=400, detail={"error": "query_not_understood", "detail": str(exc)}) from exc
        except ValueError as exc:
            raise HTTPException(status_code=500, detail={"error": "vehicle_search_failed", "detail": str(exc)}) from exc
        return {"run_id": resolved_run_id, **payload}

    @app.post("/api/video-chat")
    def video_chat(request: VideoChatRequest) -> dict[str, Any]:
        resolved_run_id = repository.resolve_run_id(request.run_id)
        if resolved_run_id is None:
            raise HTTPException(status_code=404, detail={"error": "run_not_found", "detail": "Run not found"})
        tracks_path = repository.tracks_json_path(resolved_run_id)
        if tracks_path is None or not tracks_path.exists():
            raise HTTPException(
                status_code=500,
                detail={"error": "tracks_json_missing", "detail": f"tracks.json not found for run {resolved_run_id}"},
            )
        session_id = str(request.session_id or "default").strip() or "default"
        context = chat_sessions.get(session_id, {})
        try:
            payload = handle_video_chat(
                message=request.message,
                run_id=resolved_run_id,
                tracks_path=str(tracks_path),
                repository=repository,
                session_context=context,
                llm_provider=chat_llm_provider,
            )
        except VehicleQueryParseError as exc:
            raise HTTPException(status_code=400, detail={"error": "query_not_understood", "detail": str(exc)}) from exc
        except ValueError as exc:
            raise HTTPException(status_code=500, detail={"error": "video_chat_failed", "detail": str(exc)}) from exc
        chat_sessions[session_id] = dict(payload.pop("next_context", {}) or {})
        return {"run_id": resolved_run_id, "session_id": session_id, **payload}

    @app.get("/api/filter-options")
    def get_filter_options(run_id: str | None = None) -> dict[str, Any]:
        options = repository.get_filter_options(run_id=run_id)
        return {
            "runs": ["latest", *options["runs"]],
            "cameras": options["cameras"],
            "vehicle_classes": options["vehicle_classes"],
            "colours": options["colours"],
        }

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        runs = repository.list_runs()
        current_run = runtime_state.get_current_run()
        if current_run.get("run_id"):
            runs = [
                {
                    "run_id": current_run["run_id"],
                    "status": current_run.get("pipeline_status", "running").upper(),
                    "start_time": None,
                    "completed_at": None,
                    "camera_count": len(runtime_state.list_cameras()),
                    "processed_frames": None,
                    "overall_pipeline_runtime_ms": None,
                    "run_directory": current_run.get("run_directory"),
                    "runtime": True,
                },
                *[item for item in runs if item.get("run_id") != current_run.get("run_id")],
            ]
        return runs

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        payload = repository.get_run(run_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return payload

    def _get_track_reconciliation_payload(run_id: str) -> dict[str, Any]:
        resolved_run_id = repository.resolve_run_id(run_id)
        if resolved_run_id is None:
            raise HTTPException(status_code=404, detail="Run not found")
        payload = repository.get_track_reconciliation(resolved_run_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return payload

    @app.get("/api/runs/{run_id}/track-reconciliation")
    def get_track_reconciliation(run_id: str) -> dict[str, Any]:
        return _get_track_reconciliation_payload(run_id)

    @app.get("/api/runs/{run_id}/reconciliation")
    def get_reconciliation(run_id: str) -> dict[str, Any]:
        return _get_track_reconciliation_payload(run_id)

    def _get_vehicle_identity_payload(run_id: str) -> dict[str, Any]:
        resolved_run_id = repository.resolve_run_id(run_id)
        if resolved_run_id is None:
            raise HTTPException(status_code=404, detail="Run not found")
        payload = repository.get_vehicle_identity_experiment(resolved_run_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return payload

    @app.get("/api/experimental/vehicles")
    def get_experimental_vehicles(run_id: str | None = None) -> dict[str, Any]:
        return _get_vehicle_identity_payload(run_id or "latest")

    @app.get("/api/experimental/vehicle-summary")
    def get_experimental_vehicle_summary(run_id: str | None = None) -> dict[str, Any]:
        resolved_run_id = repository.resolve_run_id(run_id)
        if resolved_run_id is None:
            raise HTTPException(status_code=404, detail="Run not found")
        payload = repository.get_vehicle_identity_summary(resolved_run_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return payload

    @app.get("/api/experimental/stationary-recovered-vehicles")
    def get_experimental_stationary_recovered_vehicles(run_id: str | None = None) -> dict[str, Any]:
        resolved_run_id = repository.resolve_run_id(run_id)
        if resolved_run_id is None:
            raise HTTPException(status_code=404, detail="Run not found")
        payload = repository.get_stationary_recovery_experiment(resolved_run_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return payload

    @app.get("/api/system/status")
    def get_system_status() -> dict[str, Any]:
        current = runtime_state.get_system_status()
        if current.get("camera_count") or current.get("pipeline_status") not in {"idle", "completed"}:
            return current
        return _persisted_system_status()

    @app.get("/api/media/{category}/{run_id}/{path:path}")
    def get_media(category: str, run_id: str, path: str) -> Any:
        parts = [item for item in str(path).split("/") if item]
        target = repository.resolve_media_path(run_id=run_id, category=category, relative_parts=parts)
        if target is None:
            raise HTTPException(status_code=404, detail="Media not found")
        return FileResponse(str(target))

    @app.websocket("/ws/live")
    async def live_updates(websocket: WebSocket) -> None:
        await websocket.accept()
        subscriber = runtime_state.subscribe()
        try:
            await websocket.send_json(
                {
                    "type": "snapshot",
                    "cameras": runtime_state.list_cameras(),
                    "system": runtime_state.get_system_status(),
                    "tracks": runtime_state.list_tracks(),
                }
            )
            while True:
                try:
                    event = subscriber.get(timeout=1.0)
                except Exception:
                    await websocket.send_json({"type": "heartbeat"})
                    continue
                await websocket.send_json(event)
        except WebSocketDisconnect:
            runtime_state.unsubscribe(subscriber)
        except Exception:
            runtime_state.unsubscribe(subscriber)
            raise

    if frontend_dist.exists():
        @app.get("/")
        def root() -> Any:
            return FileResponse(str(frontend_dist / "index.html"))

        @app.get("/{full_path:path}")
        def spa_fallback(full_path: str) -> Any:
            if full_path.startswith("api/") or full_path.startswith("ws/") or full_path.startswith("assets/"):
                raise HTTPException(status_code=404, detail="Not found")
            return FileResponse(str(frontend_dist / "index.html"))

    return app
