import { useEffect, useState } from "react";
import { fetchSystemStatus } from "../api/system";
import { SystemPanel } from "../components/common/SystemPanel";
import type { SystemStatus } from "../types/system";

export function SystemPage() {
  const [system, setSystem] = useState<SystemStatus | null>(null);

  useEffect(() => {
    void fetchSystemStatus().then(setSystem);
  }, []);

  return (
    <section className="page-stack">
      <section className="card">
        <div className="section-heading">
          <div>
            <h2>System</h2>
            <p className="muted">Real runtime and run-derived operational health metrics.</p>
          </div>
        </div>
      </section>
      <SystemPanel system={system} />
    </section>
  );
}
