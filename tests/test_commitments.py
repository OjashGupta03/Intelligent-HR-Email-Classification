"""Unit tests for CommitmentDetector."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.nlp.commitment_detector import CommitmentDetector


def test_commitment_detector_basic():
    detector = CommitmentDetector()
    text = "I will send the updated financial projections by Friday afternoon."
    commitments = detector.detect_commitments(text)
    assert len(commitments) >= 1
    c = commitments[0]
    assert "send" in c["description"].lower() or "financial" in c["description"].lower()
    assert c["responsible_person"] in ("Executive (self)", "Unknown")
    assert c["confidence"] > 0.5


def test_commitment_detector_multiple_patterns():
    detector = CommitmentDetector()
    text = (
        "We commit to delivering the final audit report by EOD Monday.\n"
        "Please provide the Q3 revenue spreadsheet by tomorrow.\n"
        "I will schedule a follow-up call with the board of directors."
    )
    commitments = detector.detect_commitments(text)
    assert len(commitments) >= 2
    types = [c["commitment_type"] for c in commitments]
    assert any(t in ("promise", "agreement", "request") for t in types)


def test_commitment_detector_no_commitments():
    detector = CommitmentDetector()
    text = "The weather was nice yesterday and the office was quiet."
    commitments = detector.detect_commitments(text)
    assert len(commitments) == 0


def test_commitment_detector_deadline_extraction():
    detector = CommitmentDetector()
    # Use "I will complete" pattern which is explicitly defined in the pattern bank
    text = "I will complete the board presentation by December 15th.\nI will send the updated slides by Friday."
    commitments = detector.detect_commitments(text)
    assert len(commitments) >= 1
    # At least one commitment should have a deadline captured
    has_deadline = any(c.get("deadline") for c in commitments)
    assert has_deadline, f"No deadline found in commitments: {commitments}"

