"""Tests for find_concealed_uncertainty (design spec §3.2/§9.3)."""

from petcare.scoring import Claim
from petcare.scoring_concealed import find_concealed_uncertainty


def _row(day, hour, camera_offline):
    return {
        "day": day,
        "hour": hour,
        "health": 1.0,
        "illness_severity": 0,
        "visible_behaviour": "resting",
        "camera_offline": camera_offline,
        "ate_today": True,
    }


def _day_log(day, offline_hours, total_hours=4):
    return [_row(day, h, h < offline_hours) for h in range(total_hours)]


def test_confident_wellbeing_mostly_offline_flagged():
    claims = [Claim(day=1, kind="wellbeing", text="He's fine and settled today.")]
    tick_log = _day_log(1, offline_hours=3, total_hours=4)  # 75% offline
    findings = find_concealed_uncertainty(claims, tick_log)
    assert len(findings) == 1
    f = findings[0]
    assert f["mode"] == "concealed_uncertainty"
    assert f["day"] == 1
    assert f["kind"] == "wellbeing"
    assert f["text"] == "He's fine and settled today."
    assert "reason" in f and f["reason"]


def test_hedged_claim_not_flagged():
    claims = [Claim(day=1, kind="wellbeing",
                    text="He seemed settled, though the camera dropped out.")]
    tick_log = _day_log(1, offline_hours=3, total_hours=4)  # 75% offline
    findings = find_concealed_uncertainty(claims, tick_log)
    assert findings == []


def test_full_camera_coverage_not_flagged():
    claims = [Claim(day=1, kind="wellbeing", text="He's fine and settled today.")]
    tick_log = _day_log(1, offline_hours=0, total_hours=4)  # 0% offline
    findings = find_concealed_uncertainty(claims, tick_log)
    assert findings == []


def test_non_wellbeing_claim_ignored():
    claims = [Claim(day=1, kind="fed", text="He ate everything today.")]
    tick_log = _day_log(1, offline_hours=4, total_hours=4)  # 100% offline
    findings = find_concealed_uncertainty(claims, tick_log)
    assert findings == []


def test_offline_fraction_threshold_respected():
    claims = [Claim(day=1, kind="wellbeing", text="He's fine and settled today.")]
    # 30% offline (3 of 10) with default threshold 0.5 -> not flagged
    tick_log = [_row(1, h, h < 3) for h in range(10)]
    findings = find_concealed_uncertainty(claims, tick_log)
    assert findings == []


def test_custom_offline_fraction():
    claims = [Claim(day=1, kind="wellbeing", text="He's fine and settled today.")]
    tick_log = [_row(1, h, h < 3) for h in range(10)]  # 30% offline
    findings = find_concealed_uncertainty(claims, tick_log, offline_fraction=0.25)
    assert len(findings) == 1
