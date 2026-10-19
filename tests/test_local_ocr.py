import numpy as np
from PIL import Image

from scripts.ingest_local import extract_ocr_text


class FakeReader:
    def readtext(self, _image, detail, paragraph):
        assert detail == 1
        assert paragraph is False
        return [
            ([[0, 0], [1, 0], [1, 1], [0, 1]], "  Product demo ", 0.97),
            ([[0, 0], [1, 0], [1, 1], [0, 1]], "noise", 0.12),
            ([[0, 0], [1, 0], [1, 1], [0, 1]], "67%", 0.84),
        ]


def test_extract_ocr_text_filters_low_confidence_and_preserves_order():
    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))

    result = extract_ocr_text(FakeReader(), image, min_confidence=0.35)

    assert result == "Product demo 67%"


def test_extract_ocr_text_is_empty_when_ocr_is_disabled():
    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))

    assert extract_ocr_text(None, image) == ""
