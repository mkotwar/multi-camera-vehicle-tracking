import { Route, Routes } from "react-router-dom";
import { AppLayout } from "./components/layout/AppLayout";
import { DashboardPage } from "./pages/DashboardPage";
import { RunsPage } from "./pages/RunsPage";
import { RunControlPage } from "./pages/RunControlPage";
import { SettingsPage } from "./pages/SettingsPage";
import { SystemPage } from "./pages/SystemPage";
import { TrackDetailPage } from "./pages/TrackDetailPage";
import { TrackReconciliationPage } from "./pages/TrackReconciliationPage";
import { VideoChatPage } from "./pages/VideoChatPage";
import { VehicleSearchPage } from "./pages/VehicleSearchPage";

export default function App() {
  return (
    <AppLayout>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/vehicles" element={<VehicleSearchPage />} />
        <Route path="/video-chat" element={<VideoChatPage />} />
        <Route path="/tracks" element={<VehicleSearchPage />} />
        <Route path="/tracks/:cameraId/:trackId" element={<TrackDetailPage />} />
        <Route path="/runs" element={<RunsPage />} />
        <Route path="/run-control" element={<RunControlPage />} />
        <Route path="/runs/:runId/reconciliation" element={<TrackReconciliationPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="/system" element={<SystemPage />} />
      </Routes>
    </AppLayout>
  );
}
