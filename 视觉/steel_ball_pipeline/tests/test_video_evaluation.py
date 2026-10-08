import pytest

from scripts.evaluate_test_videos import longest_false_run, parse_thresholds, threshold_metrics


def test_longest_false_run() -> None:
    assert longest_false_run([True, False, False, True, False]) == 2
    assert longest_false_run([False, False, False, False]) == 4
    assert longest_false_run([True, True]) == 0


def test_parse_thresholds_sorts_and_deduplicates() -> None:
    assert parse_thresholds("0.03,0.25,0.03") == (0.25, 0.03)
    with pytest.raises(ValueError):
        parse_thresholds("0.001")


def test_threshold_metrics_excludes_ignored_frames_and_checks_each_video() -> None:
    records = [
        {"video": "normal", "frame_id": 0, "evaluation_state": "ignore", "confidence": 0.0},
        {"video": "normal", "frame_id": 1, "evaluation_state": "ball", "confidence": 0.8},
        {"video": "normal", "frame_id": 2, "evaluation_state": "ball", "confidence": 0.8},
        {"video": "fast", "frame_id": 0, "evaluation_state": "ball", "confidence": 0.0},
        {"video": "fast", "frame_id": 1, "evaluation_state": "ball", "confidence": 0.8},
        {"video": "empty", "frame_id": 0, "evaluation_state": "empty", "confidence": 0.0},
    ]
    report = threshold_metrics(records, 0.25)
    assert report["normal"]["frames"] == 2
    assert report["normal"]["ignored_frames"] == 1
    assert report["normal"]["recall"] == 1.0
    assert report["fast"]["recall"] == 0.5
    assert report["passes"] is False
