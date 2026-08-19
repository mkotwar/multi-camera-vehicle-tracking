type PlateState = "readable" | "unreadable" | "missing";

export function formatDisplayPlate(text?: string | null): string | null {
  const value = String(text ?? "").trim().toUpperCase();
  return value || null;
}

export function resolvePlatePresentation({
  plateText,
  plateDetected,
  readableMissingLabel = "No plate detected",
  unreadableLabel = "Plate detected, unreadable",
}: {
  plateText?: string | null;
  plateDetected?: boolean | null;
  readableMissingLabel?: string;
  unreadableLabel?: string;
}): { label: string; state: PlateState } {
  const displayText = formatDisplayPlate(plateText);
  if (displayText) {
    return { label: displayText, state: "readable" };
  }
  if (plateDetected) {
    return { label: unreadableLabel, state: "unreadable" };
  }
  return { label: readableMissingLabel, state: "missing" };
}
