import { apiPost } from "./client";
import type { VehicleSearchRequest, VehicleSearchResponse } from "../types/vehicleSearch";

export function searchVehicles(request: VehicleSearchRequest): Promise<VehicleSearchResponse> {
  return apiPost<VehicleSearchResponse>("/api/vehicle-search", request);
}
