import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { listConfigs } from "../api/configs";
import { cancelPipelineRun, fetchDbAutoImportSetting, fetchPipelineLaunchSummary, fetchPipelineRunJob, fetchPipelineRunJobs, fetchPipelineRunLogs, startPipelineRun, updateDbAutoImportSetting } from "../api/pipelineRuns";
import type { ConfigListItem } from "../types/config";
import type { DbAutoImportSetting, PipelineLaunchSummary, PipelineRunJob, PipelineRunLogs } from "../types/pipelineRun";
import { formatRelativeDate, formatVideoTime } from "../utils/time";

const ACTIVE_STATUSES = new Set(["QUEUED", "STARTING", "RUNNING", "CANCEL_REQUESTED"]);

export function RunControlPage() {
  const [searchParams] = useSearchParams();
  const [configs, setConfigs] = useState<ConfigListItem[]>([]);
  const [selectedConfig, setSelectedConfig] = useState(searchParams.get("config") ?? "");
  const [summary, setSummary] = useState<PipelineLaunchSummary | null>(null);
  const [jobs, setJobs] = useState<PipelineRunJob[]>([]);
  const [selectedJobId, setSelectedJobId] = useState("");
  const [currentJob, setCurrentJob] = useState<PipelineRunJob | null>(null);
  const [logs, setLogs] = useState<PipelineRunLogs | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [dbAutoImport, setDbAutoImport] = useState<DbAutoImportSetting | null>(null);
  const [updatingDbAutoImport, setUpdatingDbAutoImport] = useState(false);

  useEffect(() => {
    void listConfigs()
      .then((rows) => {
        setConfigs(rows);
        if (!selectedConfig && rows[0]) {
          const preferred = rows.find((item) => item.config_name === "validation_rectangle_roi.yaml") ?? rows[0];
          setSelectedConfig(preferred.config_name);
        }
      })
      .catch((exc) => setError(errorMessage(exc)));
    void refreshJobs();
    void refreshDbAutoImport();
  }, []);

  useEffect(() => {
    if (!selectedConfig) {
      return;
    }
    void refreshLaunchSummary();
  }, [selectedConfig]);

  useEffect(() => {
    if (!selectedJobId) {
      return;
    }
    void refreshJob(selectedJobId);
  }, [selectedJobId]);

  useEffect(() => {
    if (!currentJob || !ACTIVE_STATUSES.has(currentJob.status)) {
      return;
    }
    const timer = window.setInterval(() => {
      void refreshJob(currentJob.job_id);
      void refreshJobs();
    }, 1500);
    return () => window.clearInterval(timer);
  }, [currentJob?.job_id, currentJob?.status]);

  useEffect(() => {
    const active = jobs.find((job) => ACTIVE_STATUSES.has(job.status));
    if (active && !selectedJobId) {
      setSelectedJobId(active.job_id);
      setCurrentJob(active);
    } else if (!selectedJobId && jobs[0]) {
      setSelectedJobId(jobs[0].job_id);
      setCurrentJob(jobs[0]);
    }
  }, [jobs, selectedJobId]);

  const activeJob = useMemo(() => jobs.find((job) => ACTIVE_STATUSES.has(job.status)), [jobs]);
  const canStart = Boolean(summary?.valid) && !activeJob && !starting;

  async function refreshJobs() {
    const rows = await fetchPipelineRunJobs();
    setJobs(rows);
  }

  async function refreshLaunchSummary() {
    try {
      const payload = await fetchPipelineLaunchSummary(selectedConfig);
      setSummary(payload);
    } catch (exc) {
      setSummary(null);
      setError(errorMessage(exc));
    }
  }

  async function refreshDbAutoImport() {
    try {
      setDbAutoImport(await fetchDbAutoImportSetting());
    } catch (exc) {
      setError(errorMessage(exc));
    }
  }

  async function refreshJob(jobId: string) {
    const job = await fetchPipelineRunJob(jobId);
    setCurrentJob(job);
    const logPayload = await fetchPipelineRunLogs(jobId, 180);
    setLogs(logPayload);
  }

  async function handleStartConfirmed() {
    if (!selectedConfig || starting) {
      return;
    }
    setStarting(true);
    setShowConfirm(false);
    setError(null);
    try {
      const job = await startPipelineRun(selectedConfig);
      setCurrentJob(job);
      setSelectedJobId(job.job_id);
      await refreshJobs();
    } catch (exc) {
      setError(errorMessage(exc));
    } finally {
      setStarting(false);
    }
  }

  async function handleCancel() {
    if (!currentJob) {
      return;
    }
    setError(null);
    try {
      const job = await cancelPipelineRun(currentJob.job_id);
      setCurrentJob(job);
      await refreshJobs();
      await refreshJob(job.job_id);
    } catch (exc) {
      setError(errorMessage(exc));
    }
  }

  async function handleDbAutoImportChange(enabled: boolean) {
    setUpdatingDbAutoImport(true);
    setError(null);
    try {
      const setting = await updateDbAutoImportSetting(enabled);
      setDbAutoImport(setting);
      setSummary((current) => current ? { ...current, db_import_after_run: setting.enabled } : current);
    } catch (exc) {
      setError(errorMessage(exc));
    } finally {
      setUpdatingDbAutoImport(false);
    }
  }

  return (
    <div className="page-stack run-control-page">
      <section className="settings-header">
        <div>
          <h1>Pipeline Run Control</h1>
          <p className="muted">Starts the same CLI pipeline used from PowerShell, supervised as one local job.</p>
        </div>
        <div className="settings-actions">
          <Link className="secondary-button" to={`/settings${selectedConfig ? `?config=${encodeURIComponent(selectedConfig)}` : ""}`}>Settings</Link>
          <button className="primary-button" disabled={!canStart} onClick={() => setShowConfirm(true)}>
            {starting ? "Starting..." : "Start Run"}
          </button>
        </div>
      </section>

      {error ? <div className="track-detail-state error">{error}</div> : null}

      <div className="settings-grid">
        <div className="settings-main">
          <section className="card settings-section">
            <div className="section-heading">
              <div>
                <h2>Configuration</h2>
                <p className="muted">Only saved YAML configs from the backend allowlist can be launched.</p>
              </div>
              <span className={`status ${summary?.valid ? "" : "warning"}`}>{summary?.valid ? "Valid" : "Invalid"}</span>
            </div>
            <div className="settings-toolbar">
              <label>
                <span>Config</span>
                <select value={selectedConfig} onChange={(event) => setSelectedConfig(event.target.value)}>
                  {configs.map((item) => <option key={item.config_name} value={item.config_name}>{item.config_name}</option>)}
                </select>
              </label>
            </div>
            <DbAutoImportControl setting={dbAutoImport} disabled={updatingDbAutoImport || Boolean(activeJob)} onChange={handleDbAutoImportChange} />
            {summary ? <LaunchSummary summary={summary} production={selectedConfig === "production.yaml"} /> : null}
            {summary && !summary.valid ? (
              <div className="track-detail-state error">
                {summary.errors.map((item) => <p key={`${item.path}-${item.message}`}><code>{item.path}</code>: {item.message}</p>)}
              </div>
            ) : null}
            {activeJob ? <p className="muted">A run is already active. Start is disabled until it completes or is cancelled.</p> : null}
          </section>

          <section className="card settings-section">
            <div className="section-heading">
              <h2>Current Run</h2>
              {currentJob ? <span className={`status run-status-${currentJob.status.toLowerCase()}`}>{currentJob.status}</span> : null}
            </div>
            {currentJob ? <CurrentJobCard job={currentJob} onCancel={handleCancel} /> : <div className="empty-state">No run job selected.</div>}
          </section>

          <section className="card settings-section">
            <div className="section-heading">
              <h2>Logs</h2>
              {currentJob ? <button className="secondary-button" onClick={() => refreshJob(currentJob.job_id)}>Refresh Logs</button> : null}
            </div>
            <pre className="run-log-viewer">{logs?.lines.join("\n") || "No logs loaded."}</pre>
          </section>
        </div>

        <aside className="settings-side">
          <section className="card">
            <h2>Recent Jobs</h2>
            <div className="run-job-list">
              {jobs.length === 0 ? <div className="empty-inline">No jobs yet.</div> : null}
              {jobs.map((job) => (
                <button className={job.job_id === selectedJobId ? "active" : ""} key={job.job_id} onClick={() => setSelectedJobId(job.job_id)}>
                  <strong>{job.status}</strong>
                  <span>{job.config_name}</span>
                  <code>{job.run_id ?? job.job_id}</code>
                </button>
              ))}
            </div>
          </section>
          <section className="card">
            <h2>Safety</h2>
            <p className="muted">One active GPU pipeline run is allowed. The backend launches `app.py --config config/&lt;name&gt;` with the project venv Python and no shell interpolation.</p>
          </section>
        </aside>
      </div>

      {showConfirm && summary ? (
        <div className="modal-backdrop" role="dialog" aria-modal="true">
          <section className="card confirm-modal">
            <h2>{selectedConfig === "production.yaml" ? "Start production pipeline?" : "Start pipeline run?"}</h2>
            <LaunchSummary summary={summary} production={selectedConfig === "production.yaml"} />
            <div className="settings-actions">
              <button className="secondary-button" onClick={() => setShowConfirm(false)}>Cancel</button>
              <button className="primary-button" disabled={!summary.valid || starting} onClick={handleStartConfirmed}>Confirm Start</button>
            </div>
          </section>
        </div>
      ) : null}
    </div>
  );
}

function DbAutoImportControl({ setting, disabled, onChange }: { setting: DbAutoImportSetting | null; disabled: boolean; onChange: (enabled: boolean) => void }) {
  return (
    <div className="runtime-setting-row">
      <label className="toggle-control">
        <input
          type="checkbox"
          checked={Boolean(setting?.enabled)}
          disabled={disabled || setting === null}
          onChange={(event) => onChange(event.target.checked)}
        />
        <span>DB auto-import</span>
      </label>
      <div>
        <strong>{setting?.enabled ? "On" : "Off"}</strong>
        <p className="muted">Completed runs are {setting?.enabled ? "imported into PostgreSQL automatically." : "left on disk until manually imported."}</p>
      </div>
    </div>
  );
}

function LaunchSummary({ summary, production }: { summary: PipelineLaunchSummary; production: boolean }) {
  return (
    <div className="launch-summary">
      {production ? <div className="track-detail-state error">Start production pipeline using production.yaml?</div> : null}
      <dl className="context-list">
        <div><dt>Config</dt><dd>{summary.config_name}</dd></div>
        <div><dt>Input source</dt><dd>{summary.input_sources.map((item) => `${item.camera_id}: ${item.source}`).join(" | ") || "None"}</dd></div>
        <div><dt>ROI</dt><dd>{summary.tracking_roi.enabled ? `${summary.tracking_roi.mode} / ${summary.tracking_roi.anchor}` : "disabled"}</dd></div>
        <div><dt>Plate OCR</dt><dd>{summary.plate_ocr_enabled ? "on" : "off"}</dd></div>
        <div><dt>Physical identity</dt><dd>{summary.physical_identity_enabled ? "on" : "off"}</dd></div>
        <div><dt>DB auto-import</dt><dd>{summary.db_import_after_run ? "on" : "off"}</dd></div>
      </dl>
    </div>
  );
}

function CurrentJobCard({ job, onCancel }: { job: PipelineRunJob; onCancel: () => void }) {
  const canCancel = ACTIVE_STATUSES.has(job.status);
  return (
    <div className="job-card">
      <dl className="stats-grid">
        <div><dt>Job ID</dt><dd>{job.job_id}</dd></div>
        <div><dt>Pipeline Run ID</dt><dd>{job.run_id ?? "Pending"}</dd></div>
        <div><dt>Config</dt><dd>{job.config_name}</dd></div>
        <div><dt>Stage</dt><dd>{job.current_stage}</dd></div>
        <div><dt>Processed Frames</dt><dd>{job.processed_frames ?? "Unknown"}</dd></div>
        <div><dt>Elapsed</dt><dd>{formatVideoTime(job.elapsed_seconds ?? 0)}</dd></div>
        <div><dt>PID</dt><dd>{job.pid ?? "Pending"}</dd></div>
        <div><dt>Started</dt><dd>{formatRelativeDate(job.started_at ?? job.created_at)}</dd></div>
      </dl>
      {job.error_message ? <div className="track-detail-state error">{job.error_message}</div> : null}
      <div className="run-actions">
        <button className="secondary-button" disabled={!canCancel} onClick={onCancel}>Cancel Run</button>
        {job.status === "COMPLETED" && job.run_id ? (
          <>
            <Link to={`/vehicles?run_id=${encodeURIComponent(job.run_id)}`}>Open Vehicles</Link>
            <Link to={`/video-chat?run_id=${encodeURIComponent(job.run_id)}`}>Open Video Chat</Link>
          </>
        ) : null}
      </div>
    </div>
  );
}

function errorMessage(exc: unknown): string {
  if (typeof exc === "object" && exc !== null && "detail" in exc) {
    const detail = (exc as { detail?: unknown }).detail;
    if (typeof detail === "object" && detail !== null && "detail" in detail) {
      return String((detail as { detail?: unknown }).detail);
    }
  }
  return exc instanceof Error ? exc.message : "Request failed.";
}
