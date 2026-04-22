"""In-memory patient/session history tracker for Skiavox."""

from __future__ import annotations

import csv
import io
from typing import Any, Dict, List


class PatientHistory:
    """Track scans in memory for the current app session."""

    def __init__(self) -> None:
        self._scans: List[Dict[str, Any]] = []

    def add_scan(
        self, scan_id: str, predictions: Dict[str, float], risk_score: int, timestamp: str
    ) -> None:
        """Append a scan entry to history."""
        self._scans.append(
            {
                "scan_id": scan_id,
                "predictions": predictions,
                "risk_score": risk_score,
                "timestamp": timestamp,
            }
        )

    def get_history(self) -> List[Dict[str, Any]]:
        """Return full scan history for this session."""
        return list(self._scans)

    def compare_scans(self, scan_id_1: str, scan_id_2: str) -> Dict[str, Any]:
        """Compare pathology probabilities between two stored scans."""
        s1 = next((s for s in self._scans if s["scan_id"] == scan_id_1), None)
        s2 = next((s for s in self._scans if s["scan_id"] == scan_id_2), None)
        if not s1 or not s2:
            return {"error": "One or both scan IDs not found."}

        deltas = {}
        keys = set(s1["predictions"].keys()).union(s2["predictions"].keys())
        for key in keys:
            p1 = float(s1["predictions"].get(key, 0.0))
            p2 = float(s2["predictions"].get(key, 0.0))
            deltas[key] = round(p2 - p1, 4)
        return {"scan_1": scan_id_1, "scan_2": scan_id_2, "changes": deltas}

    def get_trend(self, pathology_name: str) -> str:
        """Return improving/stable/worsening trend across session scans."""
        values = [
            float(scan["predictions"].get(pathology_name, 0.0))
            for scan in self._scans
            if pathology_name in scan["predictions"]
        ]
        if len(values) < 2:
            return "stable"
        delta = values[-1] - values[0]
        if delta > 0.05:
            return "worsening"
        if delta < -0.05:
            return "improving"
        return "stable"

    def clear_history(self) -> None:
        """Clear all session history."""
        self._scans.clear()

    def export_history_csv(self) -> str:
        """Export session history rows as CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["scan_id", "timestamp", "risk_score", "pathology", "probability"])
        for scan in self._scans:
            for pathology, prob in scan["predictions"].items():
                writer.writerow(
                    [scan["scan_id"], scan["timestamp"], scan["risk_score"], pathology, prob]
                )
        return output.getvalue()
