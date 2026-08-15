# Run Control

Run Control lets operators start and supervise the existing pipeline from the UI.

The execution path remains:

```text
UI -> FastAPI -> PipelineRunManager -> child process -> app.py -> run_pipeline()
```

The manager launches the same command used from PowerShell:

```powershell
.\.venv\Scripts\python.exe app.py --config config\<selected-config>.yaml
```

## Settings

Use the Settings page to edit and save YAML configs. YAML files in `config/` remain the canonical runtime configuration.

Secrets are not exposed through Settings or Run Control. Keep database URLs, passwords, tokens, and API keys in `.env`.

## Starting A Run

Open `Run Control`, select a saved config, review the confirmation summary, then start.

The confirmation includes:

- Config name
- Enabled input sources
- Tracking ROI status
- Plate OCR status
- Physical identity status
- `DB_IMPORT_AFTER_RUN` status

## Cancelling

Cancellation marks the job `CANCEL_REQUESTED`, terminates the subprocess, waits briefly, then force-kills the process tree on Windows if required. Partial run artifacts are retained for debugging and are not reported as completed unless the subprocess exits successfully.

## Logs

Run Control logs live under:

```text
outputs/run_jobs/<JOB_ID>/pipeline.log
```

These logs capture child process stdout/stderr and do not replace the normal pipeline log written inside `outputs/runs/<RUN_ID>/pipeline.log`.

## Job ID vs Run ID

`job_id` identifies the UI-supervised launch job.

`run_id` identifies the actual pipeline output run created by `run_pipeline()`, for example `20260815_155243`.

The UI shows `run_id` after it is detected from CLI output/logs.

## Database Import

Run Control does not reimplement database import. The child process inherits the same environment, so existing `DB_IMPORT_AFTER_RUN=true` behavior remains unchanged.

## Limitations

- One active GPU pipeline run at a time.
- Local single-machine subprocess supervision.
- Polling-based status/log updates rather than WebSocket streaming.
- If FastAPI restarts while a job is active, persisted job metadata is marked failed with a clear reattach message rather than displayed as running forever.

PowerShell execution remains supported as a fallback.
