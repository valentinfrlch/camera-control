from camera import Camera
from object_recognition import ObjectRecognizer
from pathlib import Path
import time


def test_capture_single():
    camera = Camera()
    out: Path = camera.capture_single()
    assert out.exists()
    print(f"Captured image at: {out}")


def test_live_preview():
    recognizer = ObjectRecognizer()
    camera = Camera()
    camera.live_preview(recognizer=recognizer)


def test_capture_burst():
    camera = Camera()
    out_dir: Path = camera.capture_burst(burst_count=5)
    assert out_dir.exists()
    images = list(out_dir.glob("*.nef"))
    assert len(images) == 5
    # check corresponding jpegs exist in /previews subdir
    preview_dir = out_dir.parent.parent / "previews" / out_dir.name
    assert preview_dir.exists()
    jpeg_images = list(preview_dir.glob("*.jpg"))
    assert len(jpeg_images) == 5
    print(f"Captured burst images at: {out_dir}")


def test_convert_to_jpeg():
    camera = Camera()
    nef_path = Path("tests/assets/capture-20251218-185850.nef")
    jpeg_path = camera._convert_to_jpeg(nef_path)
    assert jpeg_path.exists()
    print(f"Converted image saved at: {jpeg_path}")


def test_should_trigger_capture_requires_detection_reset():
    camera = Camera(auto_release=False)
    camera.capture_cooldown_seconds = 0.0
    capture_labels = ["Bird"]
    detections = [{"label": "Bird", "confidence": 0.9}]

    assert camera._should_trigger_capture(capture_labels, detections) is True
    # Subsequent frames that still see the same object should not retrigger captures.
    assert camera._should_trigger_capture(capture_labels, detections) is False

    # Once the object leaves the frame (no qualifying detections), the latch resets.
    assert camera._should_trigger_capture(capture_labels, []) is False

    # Seeing the object again after it disappeared should trigger another capture.
    assert camera._should_trigger_capture(capture_labels, detections) is True
