import { PointerEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { cloneConfig, getConfig, listConfigs, roiPreviewUrl, saveConfig, validateConfig } from "../api/configs";
import type { ConfigInventoryItem, ConfigListItem, ConfigValidationError, ConfigValidationResult, PipelineConfig } from "../types/config";

type RoiRectangle = {
  x_min_fraction: number;
  y_min_fraction: number;
  x_max_fraction: number;
  y_max_fraction: number;
};

type DragState =
  | { mode: "draw"; startX: number; startY: number }
  | { mode: "move"; startX: number; startY: number; original: RoiRectangle }
  | { mode: "resize"; handle: "tl" | "tr" | "bl" | "br"; startX: number; startY: number; original: RoiRectangle };

const DEFAULT_ROI: RoiRectangle = {
  x_min_fraction: 0,
  y_min_fraction: 0,
  x_max_fraction: 1,
  y_max_fraction: 1,
};

export function SettingsPage() {
  const [configs, setConfigs] = useState<ConfigListItem[]>([]);
  const [selectedConfig, setSelectedConfig] = useState("");
  const [config, setConfig] = useState<PipelineConfig | null>(null);
  const [originalConfig, setOriginalConfig] = useState<PipelineConfig | null>(null);
  const [inventory, setInventory] = useState<ConfigInventoryItem[]>([]);
  const [yamlText, setYamlText] = useState("");
  const [validation, setValidation] = useState<ConfigValidationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [newConfigName, setNewConfigName] = useState("");
  const [cameraId, setCameraId] = useState("");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [dragState, setDragState] = useState<DragState | null>(null);
  const dragStateRef = useRef<DragState | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    void listConfigs()
      .then((rows) => {
        setConfigs(rows);
        const preferred = rows.find((item) => item.config_name === "validation_rectangle_roi.yaml") ?? rows.find((item) => item.preferred) ?? rows[0];
        if (preferred) {
          setSelectedConfig(preferred.config_name);
        }
      })
      .catch((exc) => setError(errorMessage(exc)));
  }, []);

  useEffect(() => {
    if (!selectedConfig) {
      return;
    }
    void loadSelectedConfig(selectedConfig);
  }, [selectedConfig]);

  useEffect(() => {
    const onBeforeUnload = (event: BeforeUnloadEvent) => {
      if (!dirty) {
        return;
      }
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [dirty]);

  useEffect(() => {
    return () => {
      if (previewUrl?.startsWith("blob:")) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const cameras = useMemo(() => {
    const rows = getPath(config, ["input", "cameras"]);
    return Array.isArray(rows) ? rows.filter((item): item is Record<string, unknown> => isRecord(item)) : [];
  }, [config]);

  useEffect(() => {
    if (!cameraId && cameras.length > 0) {
      setCameraId(String(cameras.find((item) => item.enabled !== false)?.camera_id ?? cameras[0].camera_id ?? ""));
    }
  }, [cameraId, cameras]);

  const roi = readRoi(config);
  const validationErrors = validation?.errors ?? [];

  async function loadSelectedConfig(configName: string) {
    setError(null);
    setStatus(null);
    try {
      const detail = await getConfig(configName);
      setConfig(detail.config);
      setOriginalConfig(clone(detail.config));
      setInventory(detail.inventory);
      setYamlText(detail.yaml_text);
      setValidation(detail.validation);
      setDirty(false);
      setPreviewUrl(null);
      setPreviewError(null);
      const nextCameras = getPath(detail.config, ["input", "cameras"]);
      if (Array.isArray(nextCameras)) {
        const firstEnabled = nextCameras.find((item) => isRecord(item) && item.enabled !== false);
        const first = firstEnabled ?? nextCameras[0];
        if (isRecord(first)) {
          setCameraId(String(first.camera_id ?? ""));
        }
      }
    } catch (exc) {
      setError(errorMessage(exc));
    }
  }

  function updateValue(path: Array<string | number>, value: unknown) {
    setConfig((current) => {
      const next = setPath(current ?? {}, path, value);
      return next;
    });
    setDirty(true);
    setStatus(null);
  }

  function updateRoi(next: RoiRectangle) {
    const rounded = normalizeRoi(next);
    updateValue(["tracking_roi", "enabled"], true);
    updateValue(["tracking_roi", "mode"], "rectangle");
    updateValue(["tracking_roi", "anchor"], "bottom_center");
    updateValue(["tracking_roi", "rectangle"], rounded);
  }

  async function handleValidate() {
    if (!config || !selectedConfig) {
      return;
    }
    setStatus(null);
    setError(null);
    try {
      const result = await validateConfig(selectedConfig, config);
      setValidation(result);
      setStatus(result.valid ? "Configuration is valid." : "Configuration has validation errors.");
    } catch (exc) {
      setError(errorMessage(exc));
    }
  }

  async function handleSave() {
    if (!config || !selectedConfig) {
      return;
    }
    if (selectedConfig === "production.yaml" && !window.confirm("Save changes directly to production.yaml?")) {
      return;
    }
    setError(null);
    setStatus(null);
    try {
      const result = await saveConfig(selectedConfig, config);
      setValidation(result);
      setYamlText(result.yaml_text);
      setOriginalConfig(clone(config));
      setDirty(false);
      setStatus(`Saved ${result.config_name}. New runs will load these settings.`);
    } catch (exc) {
      setError(errorMessage(exc));
      const errors = extractValidationErrors(exc);
      if (errors.length > 0) {
        setValidation({ valid: false, errors, warnings: [] });
      }
    }
  }

  async function handleSaveAs() {
    if (!config || !selectedConfig || !newConfigName.trim()) {
      return;
    }
    setError(null);
    setStatus(null);
    try {
      const result = await cloneConfig(selectedConfig, newConfigName.trim(), config);
      setStatus(`Created ${result.config_name}.`);
      setNewConfigName("");
      const rows = await listConfigs();
      setConfigs(rows);
      setSelectedConfig(result.config_name);
    } catch (exc) {
      setError(errorMessage(exc));
    }
  }

  function handleConfigSwitch(nextName: string) {
    if (dirty && !window.confirm("Discard unsaved configuration changes?")) {
      return;
    }
    setSelectedConfig(nextName);
  }

  function handleReset() {
    if (!originalConfig) {
      return;
    }
    setConfig(clone(originalConfig));
    setDirty(false);
    setStatus("Changes reset.");
  }

  async function handleLoadFrame() {
    if (!selectedConfig || !cameraId) {
      return;
    }
    setPreviewError(null);
    try {
      const response = await fetch(`${roiPreviewUrl(selectedConfig, cameraId)}&cache=${Date.now()}`);
      if (!response.ok) {
        throw new Error(await previewErrorMessage(response));
      }
      const blob = await response.blob();
      setPreviewUrl((current) => {
        if (current?.startsWith("blob:")) {
          URL.revokeObjectURL(current);
        }
        return URL.createObjectURL(blob);
      });
    } catch (exc) {
      setPreviewUrl(null);
      setPreviewError(errorMessage(exc));
    }
  }

  function imagePoint(event: PointerEvent): { x: number; y: number } | null {
    const image = imageRef.current;
    if (!image) {
      return null;
    }
    const bounds = image.getBoundingClientRect();
    if (bounds.width <= 0 || bounds.height <= 0) {
      return null;
    }
    return {
      x: clamp((event.clientX - bounds.left) / bounds.width),
      y: clamp((event.clientY - bounds.top) / bounds.height),
    };
  }

  function handleImagePointerDown(event: PointerEvent) {
    const point = imagePoint(event);
    if (!point) {
      return;
    }
    (event.currentTarget as HTMLElement).setPointerCapture?.(event.pointerId);
    startDrag({ mode: "draw", startX: point.x, startY: point.y });
    updateRoi({
      x_min_fraction: point.x,
      y_min_fraction: point.y,
      x_max_fraction: point.x,
      y_max_fraction: point.y,
    });
  }

  function handleOverlayPointerDown(event: PointerEvent, mode: "move" | "resize", handle?: "tl" | "tr" | "bl" | "br") {
    event.stopPropagation();
    const point = imagePoint(event);
    if (!point) {
      return;
    }
    (event.currentTarget as HTMLElement).setPointerCapture?.(event.pointerId);
    startDrag(mode === "move" ? { mode: "move", startX: point.x, startY: point.y, original: roi } : { mode: "resize", handle: handle ?? "br", startX: point.x, startY: point.y, original: roi });
  }

  function handlePointerMove(event: PointerEvent) {
    const activeDrag = dragStateRef.current ?? dragState;
    if (!activeDrag) {
      return;
    }
    const point = imagePoint(event);
    if (!point) {
      return;
    }
    if (activeDrag.mode === "draw") {
      updateRoi(fromCorners(activeDrag.startX, activeDrag.startY, point.x, point.y));
      return;
    }
    if (activeDrag.mode === "move") {
      const dx = point.x - activeDrag.startX;
      const dy = point.y - activeDrag.startY;
      updateRoi(moveRoi(activeDrag.original, dx, dy));
      return;
    }
    updateRoi(resizeRoi(activeDrag.original, activeDrag.handle, point.x - activeDrag.startX, point.y - activeDrag.startY));
  }

  function startDrag(next: DragState) {
    dragStateRef.current = next;
    setDragState(next);
  }

  function endDrag() {
    dragStateRef.current = null;
    setDragState(null);
  }

  if (!config) {
    return (
      <div className="page-stack">
        <section className="card">
          <h1>Pipeline Configuration</h1>
          <p className="muted">{error ?? "Loading configuration..."}</p>
        </section>
      </div>
    );
  }

  return (
    <div className="page-stack settings-page">
      <section className="settings-header">
        <div>
          <h1>Pipeline Configuration</h1>
          <p className="muted">YAML remains the canonical runtime config. Changes apply to the next pipeline run.</p>
        </div>
        <div className="settings-actions">
          {dirty ? <span className="status warning">Unsaved changes</span> : <span className="status">Saved state</span>}
          <button className="secondary-button" onClick={handleValidate}>Validate</button>
          <button className="secondary-button" onClick={handleReset} disabled={!dirty}>Reset</button>
          <button className="primary-button" onClick={handleSave}>Save</button>
          <Link
            className={`secondary-button ${dirty || validation?.valid === false ? "disabled-link" : ""}`}
            aria-disabled={dirty || validation?.valid === false}
            to={dirty || validation?.valid === false ? "#" : `/run-control?config=${encodeURIComponent(selectedConfig)}`}
          >
            Run this configuration
          </Link>
        </div>
      </section>

      {error ? <div className="track-detail-state error">{error}</div> : null}
      {status ? <div className="empty-inline">{status}</div> : null}
      {validationErrors.length > 0 ? <ValidationPanel errors={validationErrors} /> : null}

      <section className="settings-toolbar">
        <label>
          <span>Config preset</span>
          <select value={selectedConfig} onChange={(event) => handleConfigSwitch(event.target.value)}>
            {configs.map((item) => (
              <option key={item.config_name} value={item.config_name}>{item.config_name}</option>
            ))}
          </select>
        </label>
        <label>
          <span>Save As</span>
          <input value={newConfigName} onChange={(event) => setNewConfigName(event.target.value)} placeholder="new_config.yaml" />
        </label>
        <button className="secondary-button" onClick={handleSaveAs} disabled={!newConfigName.trim()}>Save As</button>
      </section>

      <div className="settings-grid">
        <div className="settings-main">
          <ConfigSection title="Input / Camera" level="safe">
            {cameras.map((camera, index) => (
              <div className="settings-subgrid" key={String(camera.camera_id ?? index)}>
                <TextControl label="Camera ID" value={String(camera.camera_id ?? "")} onChange={(value) => updateValue(["input", "cameras", index, "camera_id"], value)} />
                <SelectControl label="Source type" value={String(camera.source_type ?? "video")} options={["video", "rtsp", "webcam"]} onChange={(value) => updateValue(["input", "cameras", index, "source_type"], value)} />
                <TextControl label="Source" value={String(camera.source ?? "")} onChange={(value) => updateValue(["input", "cameras", index, "source"], value)} />
                <ToggleControl label="Enabled" checked={Boolean(camera.enabled)} onChange={(value) => updateValue(["input", "cameras", index, "enabled"], value)} />
              </div>
            ))}
          </ConfigSection>

          <ConfigSection title="Detection" level="advanced">
            <SettingGrid>
              <TextControl label="Backend" value={stringPath(config, ["detection", "backend"])} onChange={(value) => updateValue(["detection", "backend"], value)} />
              <TextControl label="Model path" value={stringPath(config, ["detection", "model_path"])} onChange={(value) => updateValue(["detection", "model_path"], value)} />
              <NumberControl label="Confidence threshold" value={numberPath(config, ["detection", "confidence_threshold"])} step={0.01} min={0} max={1} onChange={(value) => updateValue(["detection", "confidence_threshold"], value)} />
              <NumberControl label="IOU threshold" value={numberPath(config, ["detection", "iou_threshold"])} step={0.01} min={0} max={1} onChange={(value) => updateValue(["detection", "iou_threshold"], value)} />
              <NumberControl label="Image size" value={numberPath(config, ["detection", "image_size"])} step={32} min={32} onChange={(value) => updateValue(["detection", "image_size"], value)} />
              <ToggleControl label="Agnostic NMS" checked={booleanPath(config, ["detection", "agnostic_nms"])} onChange={(value) => updateValue(["detection", "agnostic_nms"], value)} />
            </SettingGrid>
          </ConfigSection>

          <ConfigSection title="Tracking" level="advanced">
            <SettingGrid>
              <TextControl label="Backend" value={stringPath(config, ["tracking", "backend"])} onChange={(value) => updateValue(["tracking", "backend"], value)} />
              <NumberControl label="Track activation threshold" value={numberPath(config, ["tracking", "track_activation_threshold"])} step={0.01} min={0} max={1} onChange={(value) => updateValue(["tracking", "track_activation_threshold"], value)} />
              <NumberControl label="Lost track buffer" value={numberPath(config, ["tracking", "lost_track_buffer"])} step={1} min={0} onChange={(value) => updateValue(["tracking", "lost_track_buffer"], Math.round(value))} />
              <NumberControl label="Minimum matching threshold" value={numberPath(config, ["tracking", "minimum_matching_threshold"])} step={0.01} min={0} max={1} onChange={(value) => updateValue(["tracking", "minimum_matching_threshold"], value)} />
              <NumberControl label="Minimum consecutive frames" value={numberPath(config, ["tracking", "minimum_consecutive_frames"])} step={1} min={1} onChange={(value) => updateValue(["tracking", "minimum_consecutive_frames"], Math.round(value))} />
            </SettingGrid>
          </ConfigSection>

          <ConfigSection title="Tracking ROI" level="safe">
            <div className="roi-controls">
              <ToggleControl label="ROI enabled" checked={booleanPath(config, ["tracking_roi", "enabled"])} onChange={(value) => updateValue(["tracking_roi", "enabled"], value)} />
              <SelectControl label="Mode" value={stringPath(config, ["tracking_roi", "mode"]) || "rectangle"} options={["rectangle", "horizontal"]} onChange={(value) => updateValue(["tracking_roi", "mode"], value)} />
              <TextControl label="Anchor" value={stringPath(config, ["tracking_roi", "anchor"]) || "bottom_center"} onChange={(value) => updateValue(["tracking_roi", "anchor"], value)} />
              <label>
                <span>Preview camera</span>
                <select value={cameraId} onChange={(event) => setCameraId(event.target.value)}>
                  {cameras.map((camera) => (
                    <option key={String(camera.camera_id)} value={String(camera.camera_id)}>{String(camera.camera_id)}{camera.enabled === false ? " (disabled)" : ""}</option>
                  ))}
                </select>
              </label>
              <button className="secondary-button" onClick={() => void handleLoadFrame()}>Load frame</button>
            </div>
            <p className="muted">A detection is accepted when the bottom-center point of its bounding box falls inside the tracking ROI.</p>
            {previewError ? <div className="track-detail-state error">{previewError}</div> : null}
            <div
              className="roi-stage"
              onPointerDown={handleImagePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={endDrag}
              onPointerCancel={endDrag}
            >
              {previewUrl ? (
                <img ref={imageRef} src={previewUrl} alt="ROI preview frame" onError={() => setPreviewError("Preview frame could not be loaded for this camera/source.")} />
              ) : (
                <div className="roi-placeholder">Load a frame to draw the ROI</div>
              )}
              {previewUrl ? (
                <div
                  className="roi-rectangle"
                  style={roiStyle(roi)}
                  onPointerDown={(event) => handleOverlayPointerDown(event, "move")}
                >
                  {(["tl", "tr", "bl", "br"] as const).map((handle) => (
                    <button
                      aria-label={`Resize ROI ${handle}`}
                      className={`roi-handle ${handle}`}
                      key={handle}
                      onPointerDown={(event) => handleOverlayPointerDown(event, "resize", handle)}
                    />
                  ))}
                </div>
              ) : null}
            </div>
            <div className="roi-button-row">
              <button className="secondary-button" onClick={() => updateRoi(originalConfig ? readRoi(originalConfig) : DEFAULT_ROI)}>Reset ROI</button>
              <button className="secondary-button" onClick={() => updateRoi(DEFAULT_ROI)}>Full Frame</button>
              <button className="secondary-button" onClick={() => updateValue(["tracking_roi", "enabled"], false)}>Clear ROI</button>
              <button className="secondary-button" onClick={() => updateRoi(roi)}>Redraw</button>
            </div>
            <SettingGrid>
              <NumberControl label="x min" value={roi.x_min_fraction} step={0.001} min={0} max={1} onChange={(value) => updateRoi({ ...roi, x_min_fraction: value })} />
              <NumberControl label="y min" value={roi.y_min_fraction} step={0.001} min={0} max={1} onChange={(value) => updateRoi({ ...roi, y_min_fraction: value })} />
              <NumberControl label="x max" value={roi.x_max_fraction} step={0.001} min={0} max={1} onChange={(value) => updateRoi({ ...roi, x_max_fraction: value })} />
              <NumberControl label="y max" value={roi.y_max_fraction} step={0.001} min={0} max={1} onChange={(value) => updateRoi({ ...roi, y_max_fraction: value })} />
            </SettingGrid>
          </ConfigSection>

          <ConfigSection title="Evidence / Enrichment" level="safe">
            <SettingGrid>
              <ToggleControl label="Evidence enabled" checked={booleanPath(config, ["evidence", "enabled"])} onChange={(value) => updateValue(["evidence", "enabled"], value)} />
              <NumberControl label="Max candidates per track" value={numberPath(config, ["evidence", "maximum_candidates_per_track"])} step={1} min={1} onChange={(value) => updateValue(["evidence", "maximum_candidates_per_track"], Math.round(value))} />
              <ToggleControl label="Vehicle enrichment enabled" checked={booleanPath(config, ["vehicle_enrichment", "enabled"])} onChange={(value) => updateValue(["vehicle_enrichment", "enabled"], value)} />
              <NumberControl label="Best crops per track" value={numberPath(config, ["vehicle_enrichment", "best_crops_per_track"])} step={1} min={1} onChange={(value) => updateValue(["vehicle_enrichment", "best_crops_per_track"], Math.round(value))} />
              <ToggleControl label="Vehicle colour" checked={vehicleColourEnabled(config)} onChange={(value) => updateVehicleColour(config, updateValue, value)} />
              <ToggleControl label="Body type" checked={vehicleBodyTypeEnabled(config)} onChange={(value) => updateVehicleBodyType(config, updateValue, value)} />
            </SettingGrid>
          </ConfigSection>

          <ConfigSection title="Plate Detection / OCR" level="advanced">
            <SettingGrid>
              <ToggleControl label="Plate enabled" checked={plateEnabled(config)} onChange={(value) => updateValue(platePath(config, ["enabled"]), value)} />
              <ToggleControl label="Detector enabled" checked={plateDetectorEnabled(config)} onChange={(value) => updateValue(platePath(config, ["detector", "enabled"]), value)} />
              <TextControl label="Detector model path" value={stringPath(config, platePath(config, ["detector", "model_path"]))} onChange={(value) => updateValue(platePath(config, ["detector", "model_path"]), value)} />
              <ToggleControl label="OCR enabled" checked={plateOcrEnabled(config)} onChange={(value) => updateValue(platePath(config, ["ocr", "enabled"]), value)} />
              <TextControl label="OCR backend" value={String(getPath(config, platePath(config, ["ocr", "backend"])) ?? "")} onChange={(value) => updateValue(platePath(config, ["ocr", "backend"]), value)} />
            </SettingGrid>
          </ConfigSection>

          <ConfigSection title="Physical Vehicle Identity" level="advanced">
            <SettingGrid>
              <ToggleControl label="Identity enabled" checked={booleanPath(config, ["vehicle_identity", "enabled"])} onChange={(value) => updateValue(["vehicle_identity", "enabled"], value)} />
              <ToggleControl label="Conservative mode" checked={booleanPath(config, ["vehicle_identity", "conservative", "enabled"])} onChange={(value) => updateValue(["vehicle_identity", "conservative", "enabled"], value)} />
              <NumberControl label="Acceptance threshold" value={numberPath(config, ["vehicle_identity", "conservative", "acceptance_threshold"])} step={0.01} min={0} max={1} onChange={(value) => updateValue(["vehicle_identity", "conservative", "acceptance_threshold"], value)} />
              <NumberControl label="Ambiguity margin" value={numberPath(config, ["vehicle_identity", "conservative", "ambiguity_margin"])} step={0.01} min={0} max={1} onChange={(value) => updateValue(["vehicle_identity", "conservative", "ambiguity_margin"], value)} />
              <ToggleControl label="Plate assistance" checked={booleanPath(config, ["vehicle_identity", "plate_assistance", "enabled"])} onChange={(value) => updateValue(["vehicle_identity", "plate_assistance", "enabled"], value)} />
              <ToggleControl label="Plate contradiction veto" checked={booleanPath(config, ["vehicle_identity", "plate_assistance", "contradiction_veto"])} onChange={(value) => updateValue(["vehicle_identity", "plate_assistance", "contradiction_veto"], value)} />
              <ToggleControl label="Stationary recovery" checked={booleanPath(config, ["vehicle_identity", "stationary_recovery", "enabled"])} onChange={(value) => updateValue(["vehicle_identity", "stationary_recovery", "enabled"], value)} badge="Experimental" />
            </SettingGrid>
          </ConfigSection>

          <ConfigSection title="Output / Runtime" level="safe">
            <SettingGrid>
              <TextControl label="Output root directory" value={stringPath(config, ["output", "root_directory"])} onChange={(value) => updateValue(["output", "root_directory"], value)} />
              <ToggleControl label="Save run config" checked={booleanPath(config, ["output", "save_run_config"])} onChange={(value) => updateValue(["output", "save_run_config"], value)} />
              <NumberControl label="Ingestion workers" value={numberPath(config, ["ingestion", "worker_count"])} step={1} min={1} onChange={(value) => updateValue(["ingestion", "worker_count"], Math.round(value))} />
              <NumberControl label="Frame queue size" value={numberPath(config, ["ingestion", "frame_queue_size"])} step={1} min={1} onChange={(value) => updateValue(["ingestion", "frame_queue_size"], Math.round(value))} />
            </SettingGrid>
          </ConfigSection>
        </div>

        <aside className="settings-side">
          <section className="card">
            <h2>Environment Boundary</h2>
            <p className="muted">Secrets stay in .env and are not exposed here: DATABASE_URL, passwords, tokens, and API keys.</p>
            <p className="muted">Run triggering is not exposed because this API currently has no safe controlled pipeline launch endpoint.</p>
          </section>
          <section className="card">
            <h2>Config Inventory</h2>
            <div className="inventory-list">
              {inventory.slice(0, 80).map((item) => (
                <div key={item.path}>
                  <code>{item.path}</code>
                  <span className={`risk ${item.operator_level}`}>{item.operator_level}</span>
                </div>
              ))}
            </div>
          </section>
          <section className="card yaml-card">
            <h2>View YAML</h2>
            <pre>{dirty ? JSON.stringify(config, null, 2) : yamlText}</pre>
          </section>
        </aside>
      </div>
    </div>
  );
}

function ConfigSection({ title, level, children }: { title: string; level: string; children: ReactNode }) {
  return (
    <section className="card settings-section">
      <div className="section-heading">
        <h2>{title}</h2>
        <span className={`risk ${level.toLowerCase()}`}>{level}</span>
      </div>
      {children}
    </section>
  );
}

function SettingGrid({ children }: { children: ReactNode }) {
  return <div className="setting-grid">{children}</div>;
}

function TextControl({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label className="field-control">
      <span>{label}</span>
      <input value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function NumberControl({ label, value, step, min, max, onChange }: { label: string; value: number; step: number; min?: number; max?: number; onChange: (value: number) => void }) {
  return (
    <label className="field-control">
      <span>{label}</span>
      <input type="number" value={Number.isFinite(value) ? value : 0} step={step} min={min} max={max} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}

function SelectControl({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (value: string) => void }) {
  return (
    <label className="field-control">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => <option key={option} value={option}>{option}</option>)}
      </select>
    </label>
  );
}

function ToggleControl({ label, checked, onChange, badge }: { label: string; checked: boolean; onChange: (value: boolean) => void; badge?: string }) {
  return (
    <label className="toggle-control">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span>{label}</span>
      {badge ? <b>{badge}</b> : null}
    </label>
  );
}

function ValidationPanel({ errors }: { errors: ConfigValidationError[] }) {
  return (
    <section className="track-detail-state error">
      <strong>Validation failed</strong>
      <ul>
        {errors.map((error, index) => (
          <li key={`${error.path}-${index}`}><code>{error.path}</code>: {error.message}</li>
        ))}
      </ul>
    </section>
  );
}

function getPath(payload: unknown, path: Array<string | number>): unknown {
  let current = payload;
  for (const part of path) {
    if (Array.isArray(current) && typeof part === "number") {
      current = current[part];
    } else if (isRecord(current) && typeof part === "string") {
      current = current[part];
    } else {
      return undefined;
    }
  }
  return current;
}

function setPath(payload: PipelineConfig, path: Array<string | number>, value: unknown): PipelineConfig {
  const root = clone(payload);
  let current: Record<string, unknown> | unknown[] = root;
  for (let index = 0; index < path.length - 1; index += 1) {
    const part = path[index];
    const nextPart = path[index + 1];
    const existing = Array.isArray(current) ? current[Number(part)] : current[String(part)];
    const next = isRecord(existing) || Array.isArray(existing) ? clone(existing) : typeof nextPart === "number" ? [] : {};
    if (Array.isArray(current)) {
      current[Number(part)] = next;
    } else {
      current[String(part)] = next;
    }
    current = next as Record<string, unknown> | unknown[];
  }
  const last = path[path.length - 1];
  if (Array.isArray(current)) {
    current[Number(last)] = value;
  } else {
    current[String(last)] = value;
  }
  return root;
}

function readRoi(config: PipelineConfig | null): RoiRectangle {
  const raw = getPath(config, ["tracking_roi", "rectangle"]);
  if (!isRecord(raw)) {
    return DEFAULT_ROI;
  }
  return normalizeRoi({
    x_min_fraction: Number(raw.x_min_fraction ?? 0),
    y_min_fraction: Number(raw.y_min_fraction ?? 0),
    x_max_fraction: Number(raw.x_max_fraction ?? 1),
    y_max_fraction: Number(raw.y_max_fraction ?? 1),
  });
}

function normalizeRoi(roi: RoiRectangle): RoiRectangle {
  const x1 = clamp(Math.min(roi.x_min_fraction, roi.x_max_fraction));
  const y1 = clamp(Math.min(roi.y_min_fraction, roi.y_max_fraction));
  const x2 = clamp(Math.max(roi.x_min_fraction, roi.x_max_fraction));
  const y2 = clamp(Math.max(roi.y_min_fraction, roi.y_max_fraction));
  return {
    x_min_fraction: round6(x1),
    y_min_fraction: round6(y1),
    x_max_fraction: round6(Math.max(x2, x1 + 0.001)),
    y_max_fraction: round6(Math.max(y2, y1 + 0.001)),
  };
}

function roiStyle(roi: RoiRectangle) {
  return {
    left: `${roi.x_min_fraction * 100}%`,
    top: `${roi.y_min_fraction * 100}%`,
    width: `${Math.max(0.001, roi.x_max_fraction - roi.x_min_fraction) * 100}%`,
    height: `${Math.max(0.001, roi.y_max_fraction - roi.y_min_fraction) * 100}%`,
  };
}

function fromCorners(x1: number, y1: number, x2: number, y2: number): RoiRectangle {
  return normalizeRoi({ x_min_fraction: x1, y_min_fraction: y1, x_max_fraction: x2, y_max_fraction: y2 });
}

function moveRoi(roi: RoiRectangle, dx: number, dy: number): RoiRectangle {
  const width = roi.x_max_fraction - roi.x_min_fraction;
  const height = roi.y_max_fraction - roi.y_min_fraction;
  const x1 = clamp(roi.x_min_fraction + dx, 0, 1 - width);
  const y1 = clamp(roi.y_min_fraction + dy, 0, 1 - height);
  return normalizeRoi({ x_min_fraction: x1, y_min_fraction: y1, x_max_fraction: x1 + width, y_max_fraction: y1 + height });
}

function resizeRoi(roi: RoiRectangle, handle: "tl" | "tr" | "bl" | "br", dx: number, dy: number): RoiRectangle {
  const next = { ...roi };
  if (handle.includes("t")) next.y_min_fraction = roi.y_min_fraction + dy;
  if (handle.includes("b")) next.y_max_fraction = roi.y_max_fraction + dy;
  if (handle.includes("l")) next.x_min_fraction = roi.x_min_fraction + dx;
  if (handle.includes("r")) next.x_max_fraction = roi.x_max_fraction + dx;
  return normalizeRoi(next);
}

function stringPath(config: PipelineConfig, path: Array<string | number>): string {
  const value = getPath(config, path);
  return value == null ? "" : String(value);
}

function numberPath(config: PipelineConfig, path: Array<string | number>): number {
  const value = Number(getPath(config, path));
  return Number.isFinite(value) ? value : 0;
}

function booleanPath(config: PipelineConfig, path: Array<string | number>): boolean {
  return Boolean(getPath(config, path));
}

function platePath(config: PipelineConfig, suffix: string[]): Array<string | number> {
  if (isRecord(getPath(config, ["vehicle_enrichment", "enrichment", "plate"]))) {
    return ["vehicle_enrichment", "enrichment", "plate", ...suffix];
  }
  return ["vehicle_enrichment", "plate", ...suffix];
}

function plateEnabled(config: PipelineConfig): boolean {
  return Boolean(getPath(config, platePath(config, ["enabled"])) ?? getPath(config, ["vehicle_enrichment", "plate", "detection_enabled"]));
}

function plateDetectorEnabled(config: PipelineConfig): boolean {
  return Boolean(getPath(config, platePath(config, ["detector", "enabled"])) ?? getPath(config, ["vehicle_enrichment", "plate", "detection_enabled"]));
}

function plateOcrEnabled(config: PipelineConfig): boolean {
  return Boolean(getPath(config, platePath(config, ["ocr", "enabled"])) ?? getPath(config, ["vehicle_enrichment", "plate", "recognition_enabled"]));
}

function vehicleColourEnabled(config: PipelineConfig): boolean {
  return Boolean(getPath(config, ["vehicle_enrichment", "vehicle_attributes", "colour", "enabled"]) ?? getPath(config, ["vehicle_enrichment", "enrichment", "colour", "enabled"]));
}

function vehicleBodyTypeEnabled(config: PipelineConfig): boolean {
  return Boolean(getPath(config, ["vehicle_enrichment", "vehicle_attributes", "body_type", "enabled"]) ?? getPath(config, ["vehicle_enrichment", "enrichment", "body_type", "enabled"]));
}

function updateVehicleColour(config: PipelineConfig, updateValue: (path: Array<string | number>, value: unknown) => void, value: boolean) {
  updateValue(isRecord(getPath(config, ["vehicle_enrichment", "vehicle_attributes"])) ? ["vehicle_enrichment", "vehicle_attributes", "colour", "enabled"] : ["vehicle_enrichment", "enrichment", "colour", "enabled"], value);
}

function updateVehicleBodyType(config: PipelineConfig, updateValue: (path: Array<string | number>, value: unknown) => void, value: boolean) {
  updateValue(isRecord(getPath(config, ["vehicle_enrichment", "vehicle_attributes"])) ? ["vehicle_enrichment", "vehicle_attributes", "body_type", "enabled"] : ["vehicle_enrichment", "enrichment", "body_type", "enabled"], value);
}

function clamp(value: number, min = 0, max = 1): number {
  return Math.min(max, Math.max(min, Number.isFinite(value) ? value : min));
}

function round6(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function errorMessage(exc: unknown): string {
  if (isRecord(exc) && isRecord(exc.detail) && typeof exc.detail.detail === "string") {
    return exc.detail.detail;
  }
  if (exc instanceof Error) {
    return exc.message;
  }
  return "Request failed.";
}

async function previewErrorMessage(response: Response): Promise<string> {
  try {
    const payload = await response.json();
    if (isRecord(payload) && isRecord(payload.detail) && typeof payload.detail.detail === "string") {
      return payload.detail.detail;
    }
    if (isRecord(payload) && typeof payload.detail === "string") {
      return payload.detail;
    }
  } catch {
    const text = await response.text();
    if (text.trim()) {
      return text.trim();
    }
  }
  return `Preview frame request failed: ${response.status}`;
}

function extractValidationErrors(exc: unknown): ConfigValidationError[] {
  if (isRecord(exc) && isRecord(exc.detail) && Array.isArray(exc.detail.errors)) {
    return exc.detail.errors as ConfigValidationError[];
  }
  return [];
}
