import { Route, Routes } from "react-router-dom";
import { AppLayout } from "./components/layout/AppLayout";
import { DashboardPage } from "./pages/DashboardPage";
import { RunsPage } from "./pages/RunsPage";
import { SystemPage } from "./pages/SystemPage";
import { TrackDetailPage } from "./pages/TrackDetailPage";
import { VehicleSearchPage } from "./pages/VehicleSearchPage";

export default function App() {
  return (
    <AppLayout>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/vehicles" element={<VehicleSearchPage />} />
        <Route path="/tracks" element={<VehicleSearchPage />} />
        <Route path="/tracks/:cameraId/:trackId" element={<TrackDetailPage />} />
        <Route path="/runs" element={<RunsPage />} />
        <Route path="/system" element={<SystemPage />} />
      </Routes>
    </AppLayout>
  );
}
