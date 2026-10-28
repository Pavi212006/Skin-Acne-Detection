"""Offline route-level checks with mocked Flask and model dependencies.

Run with:
    python -m unittest discover -s tests -v

These tests validate upload/error handling; they do not measure the CNN's
accuracy or clinical reliability.
"""

import importlib.util
import io
import logging
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class RequestStub:
    files = {}
    url = "http://localhost/predict"


class FakeFlask:
    def __init__(self, _name):
        self.config = {}
        self.logger = logging.getLogger("skin-app-test")

    def route(self, *_args, **_kwargs):
        return lambda function: function

    def errorhandler(self, *_args, **_kwargs):
        return lambda function: function

    def run(self, *_args, **_kwargs):
        return None


class FakeModel:
    def __init__(self, output=None):
        self.output = np.asarray(output if output is not None else [[0.1, 0.2, 0.3, 0.4]], dtype=float)
        self.input_shapes = []

    def predict(self, values, **_kwargs):
        self.input_shapes.append(np.asarray(values).shape)
        return self.output


def render_template(template_name, **context):
    return {"template": template_name, **context}


def url_for(*_args, **kwargs):
    return "/static/" + kwargs.get("filename", "")


def load_app_with_stubbed_dependencies():
    flask_stub = types.ModuleType("flask")
    flask_stub.Flask = FakeFlask
    flask_stub.request = RequestStub
    flask_stub.render_template = render_template
    flask_stub.url_for = url_for

    tensorflow_stub = types.ModuleType("tensorflow")
    keras_stub = types.ModuleType("tensorflow.keras")
    models_stub = types.ModuleType("tensorflow.keras.models")
    models_stub.load_model = lambda *_args, **_kwargs: FakeModel()
    preprocessing_stub = types.ModuleType("tensorflow.keras.preprocessing")
    image_stub = types.ModuleType("tensorflow.keras.preprocessing.image")
    image_stub.img_to_array = lambda image: np.asarray(image)
    keras_stub.models = models_stub
    keras_stub.preprocessing = preprocessing_stub
    preprocessing_stub.image = image_stub
    tensorflow_stub.keras = keras_stub

    modules = {
        "flask": flask_stub,
        "tensorflow": tensorflow_stub,
        "tensorflow.keras": keras_stub,
        "tensorflow.keras.models": models_stub,
        "tensorflow.keras.preprocessing": preprocessing_stub,
        "tensorflow.keras.preprocessing.image": image_stub,
    }
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location("skin_app_under_test", ROOT / "app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class UploadedFile:
    def __init__(self, filename, content):
        self.filename = filename
        self.stream = io.BytesIO(content)


def valid_png_bytes():
    output = io.BytesIO()
    Image.new("RGB", (200, 100), (30, 80, 120)).save(output, format="PNG")
    return output.getvalue()


class UploadValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_module = load_app_with_stubbed_dependencies()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.app_module.UPLOAD_FOLDER = Path(self.temp_dir.name) / "uploads"
        self.app_module.app.config["UPLOAD_FOLDER"] = str(self.app_module.UPLOAD_FOLDER)
        self.app_module.model = FakeModel()
        RequestStub.files = {}

    def test_valid_image_is_resized_and_sent_as_rgb_batch(self):
        RequestStub.files = {"file": UploadedFile("skin.png", valid_png_bytes())}
        response = self.app_module.predict()
        self.assertIsNotNone(response)
        self.assertEqual(self.app_module.model.input_shapes[-1], (1, 126, 126, 3))

    def test_unsupported_extension_is_rejected(self):
        RequestStub.files = {"file": UploadedFile("skin.gif", valid_png_bytes())}
        response = self.app_module.predict()
        self.assertIsInstance(response, tuple)
        self.assertEqual(response[1], 400)

    def test_corrupt_image_is_rejected(self):
        RequestStub.files = {"file": UploadedFile("broken.png", b"not an image")}
        response = self.app_module.predict()
        self.assertIsInstance(response, tuple)
        self.assertEqual(response[1], 400)

    def test_empty_filename_is_rejected(self):
        RequestStub.files = {"file": UploadedFile("", valid_png_bytes())}
        response = self.app_module.predict()
        self.assertIsInstance(response, tuple)
        self.assertEqual(response[1], 400)

    def test_unexpected_model_output_shape_is_rejected(self):
        self.app_module.model = FakeModel([[0.1, 0.2, 0.3, 0.2, 0.1]])
        RequestStub.files = {"file": UploadedFile("skin.png", valid_png_bytes())}
        response = self.app_module.predict()
        self.assertIsInstance(response, tuple)
        self.assertEqual(response[1], 500)

    def test_non_finite_model_scores_are_rejected(self):
        self.app_module.model = FakeModel([[float("nan"), 0.2, 0.3, 0.4]])
        RequestStub.files = {"file": UploadedFile("skin.png", valid_png_bytes())}
        response = self.app_module.predict()
        self.assertIsInstance(response, tuple)
        self.assertEqual(response[1], 500)

    def test_original_filename_is_not_used_as_saved_path(self):
        RequestStub.files = {"file": UploadedFile("../../unsafe.png", valid_png_bytes())}
        response = self.app_module.predict()
        self.assertIsNotNone(response)
        saved_files = list(self.app_module.UPLOAD_FOLDER.glob("*"))
        self.assertEqual(len(saved_files), 1)
        self.assertTrue(saved_files[0].is_file())
        self.assertNotIn("..", saved_files[0].name)
        self.assertNotIn("/", saved_files[0].name)
        self.assertEqual(saved_files[0].suffix, ".jpg")

    def test_five_megabyte_request_limit_is_configured(self):
        self.assertEqual(self.app_module.app.config["MAX_CONTENT_LENGTH"], 5 * 1024 * 1024)

    def test_missing_file_is_rejected(self):
        RequestStub.files = {}
        response = self.app_module.predict()
        self.assertIsInstance(response, tuple)
        self.assertEqual(response[1], 400)


if __name__ == "__main__":
    unittest.main()
