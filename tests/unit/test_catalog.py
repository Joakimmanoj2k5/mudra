from utils.common.gesture_catalog import ALPHABETS, WORD_SPECS


def test_catalog_counts():
    assert len(ALPHABETS) == 26
    assert 50 <= len(WORD_SPECS) <= 100
