"""Fetch the frozen public model bundle when a hosted checkout has no models."""
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

BUNDLE_URL = "https://drive.usercontent.google.com/download?id=19T-_vRUf9mgBcDSPEE0nPbmtNckRkXmX&export=download&confirm=t"
BUNDLE_SHA256 = "d511654e31fc394339764a5e3dfecad47374c58ec811da85d576530b95c31d96"
FILES = (
    "m2_silver_distilroberta/config.json", "m2_silver_distilroberta/merges.txt",
    "m2_silver_distilroberta/model.safetensors", "m2_silver_distilroberta/special_tokens_map.json",
    "m2_silver_distilroberta/tokenizer.json", "m2_silver_distilroberta/tokenizer_config.json",
    "m2_silver_distilroberta/vocab.json", "m2_silver_threshold.json",
    "types_silver.joblib", "m1_is_need.joblib", "m1_types.joblib", "m1_threshold.json",
)


def install_bundle(archive, models_dir):
    """Verify the frozen bundle and extract only known model files."""
    archive, models_dir = Path(archive), Path(models_dir)
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != BUNDLE_SHA256:
        raise ValueError("Model bundle checksum differs from the verified frozen release.")
    with zipfile.ZipFile(archive) as bundle:
        for name in FILES:
            if "models/" + name not in bundle.namelist():
                raise ValueError(f"Missing model file: {name}")
        for name in FILES:
            destination = models_dir / name
            if destination.is_file():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open("models/" + name) as source, tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as target:
                temporary_path = Path(target.name)
                shutil.copyfileobj(source, target)
            os.replace(temporary_path, destination)


def ensure_models(models_dir):
    models_dir = Path(models_dir)
    if all((models_dir / name).is_file() for name in FILES):
        return
    with tempfile.TemporaryDirectory(prefix="helproute-models-") as directory:
        archive = Path(directory) / "bundle.zip"
        request = urllib.request.Request(BUNDLE_URL, headers={"User-Agent": "HelpRoute/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as target:
            shutil.copyfileobj(response, target, length=1024 * 1024)
        install_bundle(archive, models_dir)
