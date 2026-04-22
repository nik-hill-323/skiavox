"""Second-opinion consensus engine for ensemble outputs."""

from __future__ import annotations

from typing import Any, Dict, List


class SecondOpinionEngine:
    """Analyze consistency and uncertainty across model outputs."""

    def generate_second_opinion(
        self,
        predictions: Dict[str, float],
        confidence_intervals: Dict[str, float],
        agreement_score: float,
    ) -> Dict[str, Any]:
        """Generate structured second-opinion report data."""
        consensus_findings: List[str] = []
        disputed_findings: List[str] = []
        high_confidence_findings: List[str] = []
        uncertain_findings: List[str] = []

        for pathology, prob in predictions.items():
            ci = confidence_intervals.get(pathology, 0.0)
            if prob >= 0.5 and agreement_score >= 70:
                consensus_findings.append(pathology)
            if ci > 0.2:
                disputed_findings.append(pathology)
                uncertain_findings.append(pathology)
            if ci < 0.1 and prob >= 0.4:
                high_confidence_findings.append(pathology)

        if agreement_score >= 80:
            level = "STRONG"
            recommendation = "Findings consistent across models"
        elif agreement_score >= 60:
            level = "MODERATE"
            recommendation = "Moderate model concordance; radiologist confirmation advised"
        else:
            level = "WEAK"
            recommendation = (
                "Significant model disagreement - radiologist review strongly recommended"
            )

        text = (
            "SECOND OPINION SUMMARY\n"
            f"- Agreement level: {level} ({agreement_score:.1f}%)\n"
            f"- Consensus findings: {', '.join(consensus_findings) if consensus_findings else 'None'}\n"
            f"- Disputed findings: {', '.join(disputed_findings) if disputed_findings else 'None'}\n"
            f"- High-confidence findings: {', '.join(high_confidence_findings) if high_confidence_findings else 'None'}\n"
            f"- Uncertain findings: {', '.join(uncertain_findings) if uncertain_findings else 'None'}\n"
            f"- Recommendation: {recommendation}"
        )

        return {
            "consensus_findings": consensus_findings,
            "disputed_findings": disputed_findings,
            "high_confidence_findings": high_confidence_findings,
            "uncertain_findings": uncertain_findings,
            "recommendation": recommendation,
            "agreement_level": level,
            "second_opinion_text": text,
        }
