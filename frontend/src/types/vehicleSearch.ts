export type VehicleSearchRequest = {
  query: string;
  run_id?: string;
};

export type VehicleQuery = {
  vehicle_class: string | null;
  colour: string | null;
  start_time: number | null;
  end_time: number | null;
  camera_id: string | null;
  intent: "COUNT" | "LIST" | "SUMMARY" | "UNIQUE_CLASSES" | "UNIQUE_COLOURS";
};

export type VehicleAnalyticsResult = {
  total?: number;
  by_class?: Record<string, number>;
  by_colour?: Record<string, number>;
  vehicle_ids?: string[];
  vehicle_classes_present?: string[];
  colours_present?: string[];
  total_unique_vehicles?: number;
  vehicle_classes?: Record<string, number>;
  colours?: Record<string, number>;
};

export type VehicleSearchResponse = {
  run_id: string;
  original_query: string;
  parsed_query: VehicleQuery;
  analytics_result: VehicleAnalyticsResult;
  response: string;
};
