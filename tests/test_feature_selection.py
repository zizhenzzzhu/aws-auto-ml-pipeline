import json
from pathlib import Path

from stages.feature_engineering.feature_select import main


def test_feature_selection_gate_passes_with_selected_features(tmp_path, monkeypatch):
    feature_log = tmp_path / "feature_log.json"
    feature_log.write_text(json.dumps({"selected_features": ["f1"], "generated_feature_count": 3}), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        ["feature_select.py", "--feature-log-path", str(feature_log), "--min-selected-features", "1"],
    )
    main()


def test_feature_selection_gate_fails_without_selected_features(tmp_path, monkeypatch):
    feature_log = tmp_path / "feature_log.json"
    feature_log.write_text(json.dumps({"selected_features": []}), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        ["feature_select.py", "--feature-log-path", str(feature_log), "--min-selected-features", "1"],
    )

    try:
        main()
    except AssertionError as exc:
        assert "Too few selected features" in str(exc)
    else:
        raise AssertionError("Expected feature selection gate to fail.")
