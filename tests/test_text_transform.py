import pytest
from src.data.text_transform import TextTransform

def test_text_transform_basic():
    tt = TextTransform()
    text = "mepawokyew"
    encoded = tt.text_to_int(text)
    assert len(encoded) == len(text)
    decoded = tt.int_to_text(encoded)
    assert decoded == text

def test_text_transform_akan_chars():
    tt = TextTransform()
    text = "ɛdeɛn na ɛrekɔso"
    encoded = tt.text_to_int(text)
    decoded = tt.int_to_text(encoded)
    assert "ɛ" in decoded
    assert "ɔ" in decoded
    assert decoded == text


def test_digits_survive_encoding():
    """Health-corpus transcripts write dosages numerically; dropping them would
    leave the audio saying a number the label does not contain."""
    tt = TextTransform()
    text = "fa 2 anɔpa na 4 anwummerɛ"
    assert tt.int_to_text(tt.text_to_int(text)) == text


def test_punctuation_is_still_dropped():
    tt = TextTransform()
    assert tt.int_to_text(tt.text_to_int("mepawokyew, ɔdɔ!")) == "mepawokyew ɔdɔ"


def test_vocab_size_matches_blank_index():
    tt = TextTransform()
    assert tt.vocab_size == 41
    assert tt.blank_label == tt.vocab_size - 1

    without = TextTransform(include_digits=False)
    assert without.vocab_size == 31
    # Letter indices must not shift when digits are toggled off.
    assert all(without.char_map[c] == tt.char_map[c] for c in "abzɛɔ'")
