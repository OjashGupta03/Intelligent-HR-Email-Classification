"""Unit tests for RiskDetector."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.nlp.risk_detector import RiskDetector


def test_financial_risk_detection():
    detector = RiskDetector()
    text = "We are anticipating a severe revenue shortfall and budget overrun in Q4."
    risks = detector.detect_risks(text)
    categories = [r["category"] for r in risks]
    assert "financial" in categories
    fin_risk = next(r for r in risks if r["category"] == "financial")
    assert fin_risk["severity"] in ("medium", "high", "critical")
    assert "financial" in fin_risk["suggested_action"].lower() or len(fin_risk["suggested_action"]) > 0


def test_legal_compliance_risk():
    detector = RiskDetector()
    text = "We just received a subpoena regarding an SEC investigation and potential breach of contract."
    risks = detector.detect_risks(text)
    categories = [r["category"] for r in risks]
    assert "legal_compliance" in categories or "contract" in categories


def test_security_and_operational_risk():
    detector = RiskDetector()
    text = "There has been a data breach affecting our production database causing a complete system outage."
    risks = detector.detect_risks(text)
    categories = [r["category"] for r in risks]
    assert "security" in categories or "operational" in categories


def test_customer_dissatisfaction_and_escalation():
    detector = RiskDetector()
    text = "Our tier-1 enterprise client is threatening to cancel the contract and churn immediately due to poor service."
    risks = detector.detect_risks(text)
    categories = [r["category"] for r in risks]
    assert any(c in categories for c in ("customer_dissatisfaction", "contract", "escalation"))


def test_no_risk_detection():
    detector = RiskDetector()
    text = "Hope you have a fantastic weekend with your family! Looking forward to our casual lunch."
    risks = detector.detect_risks(text)
    assert len(risks) == 0

