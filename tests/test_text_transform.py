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
