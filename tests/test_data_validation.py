from argparse import Namespace

from stages.validation.data_validation_spark import validate_class_balance


class Row(dict):
    def __getattr__(self, key):
        return self[key]


class FakeGroupedFrame:
    def __init__(self, rows):
        self.rows = rows

    def count(self):
        return self

    def collect(self):
        return self.rows


class FakeFrame:
    def __init__(self, rows):
        self.rows = rows

    def groupBy(self, _column):
        return FakeGroupedFrame(self.rows)


def test_validate_class_balance_passes_when_minority_ratio_is_large_enough():
    frame = FakeFrame([Row(label=0, count=40), Row(label=1, count=60)])
    result = validate_class_balance(frame, "label", 0.05)
    assert result["minority_ratio"] == 0.4


def test_validate_class_balance_fails_when_minority_ratio_is_too_small():
    frame = FakeFrame([Row(label=0, count=1), Row(label=1, count=99)])
    try:
        validate_class_balance(frame, "label", 0.05)
    except AssertionError as exc:
        assert "Severe class imbalance" in str(exc)
    else:
        raise AssertionError("Expected severe imbalance failure.")
