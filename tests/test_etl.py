from stages.etl.etl_spark import normalize_name, unique_names


def test_normalize_name_removes_special_characters():
    assert normalize_name(" Feature 1 (%) ") == "feature_1"


def test_unique_names_suffixes_duplicates():
    assert unique_names(["A", "a", "A!"]) == ["a", "a_2", "a_3"]
