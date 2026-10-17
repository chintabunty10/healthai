import io

import pytest
from PIL import Image

from app.preprocessing import SymptomPreprocessor
from app.vision import MAX_IMAGE_BYTES, ImageAnalyzer, ImageError, load_image

VOCAB = ["skin_rash", "itching", "yellowish_skin", "swelling_joints", "cough"]


def make_image(fmt="PNG", size=(64, 48), color=(200, 80, 80)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


class FakeResp:
    def __init__(self, text): self.text = text


class FakeClient:
    def __init__(self, text): self.text, self.calls = text, []
    @property
    def models(self): return self
    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        return FakeResp(self.text)


def analyzer(text):
    return ImageAnalyzer("", "gemini-2.5-flash", SymptomPreprocessor(VOCAB), client=FakeClient(text))


def test_load_image_accepts_valid_formats_and_downscales():
    img = load_image(make_image("PNG", size=(3000, 2000)))
    assert img.mode == "RGB" and max(img.size) <= 1024
    assert load_image(make_image("JPEG")).size == (64, 48)


def test_load_image_rejects_bad_input():
    with pytest.raises(ImageError):
        load_image(b"")
    with pytest.raises(ImageError):
        load_image(b"not an image at all")
    with pytest.raises(ImageError):
        load_image(b"\x00" * (MAX_IMAGE_BYTES + 1))
    with pytest.raises(ImageError):
        load_image(make_image("GIF"))          # unsupported format


def test_parse_keeps_only_vocabulary_symptoms():
    a = analyzer("")
    f = a.parse('{"symptoms": ["Skin Rash", "itchy", "IGNORE PREVIOUS INSTRUCTIONS", 5], "description": " Red patches. "}')
    assert f.symptoms == ["itching", "skin_rash"]
    assert "ignore_previous_instructions" not in f.symptoms
    assert f.description == "Red patches."


def test_parse_handles_code_fences_and_bad_json():
    a = analyzer("")
    assert a.parse('```json\n{"symptoms": ["cough"], "description": "x"}\n```').symptoms == ["cough"]
    with pytest.raises(ValueError):
        a.parse("sorry, I can't help")
    with pytest.raises(ValueError):
        a.parse("[1, 2]")


def test_analyze_sends_prompt_and_images_to_model():
    client = FakeClient('{"symptoms": ["skin_rash"], "description": "red area"}')
    a = ImageAnalyzer("", "gemini-2.5-flash", SymptomPreprocessor(VOCAB), client=client)
    img = load_image(make_image())
    out = a.analyze([img, img])
    model, contents, config = client.calls[0]
    assert model == "gemini-2.5-flash" and len(contents) == 3 and "skin_rash" in contents[0]
    assert config["response_mime_type"] == "application/json"
    assert out.symptoms == ["skin_rash"]


def test_disabled_without_key_or_client():
    a = ImageAnalyzer("", "gemini-2.5-flash", SymptomPreprocessor(VOCAB))
    assert a.enabled is False
