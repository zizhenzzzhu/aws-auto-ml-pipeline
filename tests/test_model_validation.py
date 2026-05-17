import json

from stages.validation.model_validation import load_champion_auc


def test_load_champion_auc_prefers_explicit_value():
    args = type("Args", (), {"champion_auc": 0.81})()
    assert load_champion_auc(args, "unused", 128) == 0.81


def test_load_champion_auc_reads_metrics_file(tmp_path):
    metrics = tmp_path / "champion_metrics.json"
    metrics.write_text(json.dumps({"auc": 0.79}), encoding="utf-8")

    args = type(
        "Args",
        (),
        {
            "champion_auc": None,
            "champion_metrics_file": str(metrics),
            "champion_model_package_arn": None,
            "champion_model_dir": None,
            "region": "us-west-2",
        },
    )()

    assert load_champion_auc(args, "unused", 128) == 0.79
