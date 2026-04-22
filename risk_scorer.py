"""Clinical risk scoring engine for Skiavox."""

from __future__ import annotations

from typing import Any, Dict, Optional


class RiskScorer:
    """Compute aggregate risk from pathology probabilities and demographics."""

    URGENCY_POINTS = {"CRITICAL": 40, "HIGH": 20, "MODERATE": 10, "LOW": 5}

    def __init__(self, urgency_map: Dict[str, str], thresholds: Dict[str, float]) -> None:
        self.urgency_map = urgency_map
        self.thresholds = thresholds

    def calculate_risk_score(
        self,
        predictions: Dict[str, float],
        patient_age: Optional[int] = None,
        patient_sex: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return comprehensive risk assessment object."""
        raw_score = 0.0
        concerns = []
        risk_breakdown: Dict[str, Dict[str, Any]] = {}

        for pathology, prob in predictions.items():
            urgency = self.urgency_map.get(pathology, "LOW")
            threshold = self.thresholds.get(pathology, 0.5)
            triggered = prob >= threshold
            points = self.URGENCY_POINTS.get(urgency, 5) if triggered else 0
            raw_score += points
            if triggered:
                concerns.append((pathology, prob, urgency))
            risk_breakdown[pathology] = {
                "probability": round(prob, 4),
                "threshold": threshold,
                "triggered": triggered,
                "urgency": urgency,
                "points": points,
            }

        if patient_age is not None and patient_age > 60:
            raw_score *= 1.2

        risk_score = int(min(raw_score, 100))

        if any(c[2] == "CRITICAL" for c in concerns):
            overall_risk, risk_color = "CRITICAL", "red"
        elif risk_score >= 60:
            overall_risk, risk_color = "HIGH", "orange"
        elif risk_score >= 30:
            overall_risk, risk_color = "MODERATE", "yellow"
        else:
            overall_risk, risk_color = "LOW", "green"

        sorted_concerns = sorted(concerns, key=lambda x: x[1], reverse=True)
        primary = sorted_concerns[0][0] if sorted_concerns else "No acute concern"
        secondary = [c[0] for c in sorted_concerns[1:4]]
        immediate = overall_risk == "CRITICAL"

        if immediate:
            followup = "Immediate physician/radiologist escalation required."
        elif overall_risk == "HIGH":
            followup = "Urgent specialist review within 24 hours recommended."
        elif overall_risk == "MODERATE":
            followup = "Clinical follow-up and correlation with symptoms recommended."
        else:
            followup = "Routine follow-up as clinically indicated."

        if patient_sex:
            followup += f" (Patient sex recorded: {patient_sex})"

        return {
            "overall_risk": overall_risk,
            "risk_score": risk_score,
            "risk_color": risk_color,
            "primary_concern": primary,
            "secondary_concerns": secondary,
            "immediate_action_required": immediate,
            "recommended_followup": followup,
            "risk_breakdown": risk_breakdown,
        }
