"""
End-to-end OCR pipeline tests.

Each test feeds a real card photo through: detect → perspective warp → OCR →
parse. Asserts the OCR+parse can recover the card's identity (set code +
collector number). Card name lookup via Scryfall is tested separately to
avoid making the core pipeline tests network-dependent.

Run:  python3 -m pytest tests/test_ocr_pipeline.py -v
"""
from __future__ import annotations

import os
import pytest

# Import the pipeline under test (will be created next)
from card_pipeline import scan_card_image

IMAGES = os.path.join(os.path.dirname(__file__), '..', 'images')


# (image_file, expected_name_substring_lowercase, expected_set, expected_collector)
CARD_CASES = [
    ('Excava-SOC-0002.png',   'excava',              'soc', '2'),
    ('Gandalf-LTR-0322.jpg',  'gandalf the grey',    'ltr', '322'),
    ('IMG_0446.jpg',          'mishra, tamer',       'bro', '217'),
    ('IMG_0447.jpg',          'swiftfoot boots',     'bro', '58'),
    ('IMG_0448.jpg',          "tocasia's dig site",  'bro', '266'),
    ('image0.jpg',            'gandalf, friend',     'ltr', '308'),
]


@pytest.mark.parametrize('image_file,expected_name,expected_set,expected_collector', CARD_CASES)
def test_scan_identifies_set_and_collector(image_file, expected_name, expected_set, expected_collector):
    """Pipeline recovers at minimum the set code + collector number."""
    path = os.path.join(IMAGES, image_file)
    result = scan_card_image(path)

    assert result is not None, f"No detection for {image_file}"
    assert result.set_code is not None, f"Set code not parsed for {image_file} (blText={result.bl_text!r})"
    assert result.collector_number is not None, f"Collector number not parsed for {image_file} (blText={result.bl_text!r})"
    assert result.set_code.lower() == expected_set, \
        f"{image_file}: expected set {expected_set!r}, got {result.set_code!r} (blText={result.bl_text!r})"
    assert str(int(result.collector_number)) == str(int(expected_collector)), \
        f"{image_file}: expected collector {expected_collector!r}, got {result.collector_number!r} (blText={result.bl_text!r})"


@pytest.mark.parametrize('image_file,expected_name,expected_set,expected_collector', CARD_CASES)
def test_scan_recovers_card_name_via_scryfall(image_file, expected_name, expected_set, expected_collector):
    """Bonus: Scryfall lookup by set+collector returns the right card."""
    path = os.path.join(IMAGES, image_file)
    result = scan_card_image(path, query_scryfall=True)

    assert result is not None
    assert result.card_name is not None, f"Scryfall did not return a card for {image_file}"
    assert expected_name.lower() in result.card_name.lower(), \
        f"{image_file}: expected name containing {expected_name!r}, got {result.card_name!r}"
