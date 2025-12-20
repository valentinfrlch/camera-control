import atexit
import queue
import threading
from typing import Dict, Generator, List, Optional

import cv2
from flask import Flask, Response, jsonify, render_template, request, url_for

from camera import Camera
from object_recognition import ObjectRecognizer

app = Flask(__name__)

_MJPEG_BOUNDARY = "frame"
_FRAME_TIMEOUT_SECONDS = 3.0
_JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

_stream_lock = threading.Lock()
_stream_proc = None
_stream_thread = None
_frame_queue: Optional["queue.Queue"] = None
_recognizer: Optional[ObjectRecognizer] = None
_auto_capture_enabled = False
_capture_labels: List[str] = []
_camera = Camera(capture_confidence_threshold=0.6)

_SHOOT_MODE_SINGLE = "single"
_SHOOT_MODE_BURST = "burst"
_MIN_BURST_COUNT = 2
_MAX_BURST_COUNT = 10
_DEFAULT_BURST_COUNT = 5

_shoot_mode = _SHOOT_MODE_BURST
_burst_count = _DEFAULT_BURST_COUNT

_MODE_AUTO = "auto"
_MODE_MANUAL = "manual"
_MODE_TIMELAPSE = "timelapse"
_VALID_UI_MODES = {_MODE_AUTO, _MODE_MANUAL, _MODE_TIMELAPSE}
_active_ui_mode = _MODE_AUTO


def _init_recognizer() -> Optional[ObjectRecognizer]:
    try:
        return ObjectRecognizer()
    except Exception as exc:
        print(f"Object recognition unavailable, running without it: {exc}")
        return None


_recognizer = _init_recognizer()


def _available_labels() -> List[str]:
    if _recognizer is None:
        return []
    names = getattr(_recognizer, "class_names", {})
    if isinstance(names, dict):
        return [names[idx] for idx in sorted(names.keys())]
    if isinstance(names, list):
        return names
    return list(names)


def _capture_settings_payload() -> Dict[str, object]:
    return {"mode": _shoot_mode, "burst_count": _burst_count}


def _shutdown_stream() -> None:
    global _stream_proc, _frame_queue
    proc = _stream_proc
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()
    _stream_proc = None
    _frame_queue = None


def _ensure_stream_queue() -> Optional["queue.Queue"]:
    global _stream_proc, _stream_thread, _frame_queue
    if (
        _frame_queue is not None
        and _stream_proc is not None
        and _stream_proc.poll() is None
    ):
        return _frame_queue

    with _stream_lock:
        if (
            _frame_queue is None
            or _stream_proc is None
            or _stream_proc.poll() is not None
        ):
            _shutdown_stream()
            try:
                if _camera is None:
                    raise RuntimeError("Camera is not initialized")
                _stream_proc, _stream_thread, _frame_queue = (
                    _camera.start_live_preview_stream()
                )
            except Exception as exc:
                print(f"Unable to start live preview stream: {exc}")
                _frame_queue = None
    return _frame_queue


def _handle_capture_request(detections: Optional[List[Dict[str, object]]]) -> None:
    global _camera, _recognizer, _stream_proc, _stream_thread, _frame_queue
    global _shoot_mode, _burst_count
    if (
        _camera is None
        or _recognizer is None
        or not _auto_capture_enabled
        or _active_ui_mode != _MODE_AUTO
    ):
        return
    capture_labels = _capture_labels or getattr(_recognizer, "capture_objects", None)
    if not capture_labels or not detections:
        return
    if not _camera._should_trigger_capture(capture_labels, detections):
        return

    with _stream_lock:
        _camera._clear_frame_queue(_frame_queue)
        _camera._debug_log_queue("webserver", _frame_queue)
        try:
            _stream_proc, _stream_thread, _frame_queue = (
                _camera._restart_preview_with_capture(
                    _stream_proc,
                    _stream_thread,
                    _frame_queue,
                    shoot_mode=_shoot_mode,
                    burst_count=_burst_count,
                )
            )
        except Exception as capture_error:
            print(f"Unable to complete capture workflow: {capture_error}")
            _stream_proc = None
            _stream_thread = None
            _frame_queue = None


def _apply_recognition(frame):
    global _recognizer
    if _recognizer is None or _active_ui_mode != _MODE_AUTO:
        return frame
    try:
        annotated, detections = _recognizer.annotate(frame)
        _handle_capture_request(detections)
        return annotated
    except Exception as exc:
        print(f"Disabling object recognition due to inference error: {exc}")
        _recognizer = None
        return frame


def _next_frame_bytes() -> Optional[bytes]:
    q = _ensure_stream_queue()
    if q is None:
        return None
    try:
        frame = q.get(timeout=_FRAME_TIMEOUT_SECONDS)
    except queue.Empty:
        return None

    annotated = _apply_recognition(frame)
    success, buffer = cv2.imencode(".jpg", annotated, _JPEG_PARAMS)
    if not success:
        return None
    return buffer.tobytes()


def _mjpeg_stream() -> Generator[bytes, None, None]:
    boundary = _MJPEG_BOUNDARY.encode()
    while True:
        frame_bytes = _next_frame_bytes()
        if frame_bytes is None:
            continue
        yield (
            b"--"
            + boundary
            + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
            + str(len(frame_bytes)).encode()
            + b"\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


@app.route("/")
def index() -> str:
    return render_template(
        "index.html",
        stream_url=url_for("stream"),
        health_url=url_for("health"),
        labels_url=url_for("api_labels"),
        capture_mode_url=url_for("api_capture_mode"),
        capture_label_url=url_for("api_capture_label"),
        capture_settings_url=url_for("api_capture_settings"),
        ui_mode_url=url_for("api_ui_mode"),
        manual_capture_url=url_for("api_manual_capture"),
    )


@app.route("/manual")
def manual() -> str:
    return render_template(
        "manual.html",
        stream_url=url_for("stream"),
        health_url=url_for("health"),
    )


@app.route("/timelapse")
def timelapse() -> str:
    return render_template(
        "timelapse.html",
        stream_url=url_for("stream"),
        health_url=url_for("health"),
    )


@app.route("/stream")
def stream() -> Response:
    return Response(
        _mjpeg_stream(),
        mimetype=f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY}",
    )


@app.route("/api/health")
def health():
    stream_ready = _stream_proc is not None and _stream_proc.poll() is None
    return jsonify({"status": "ok" if stream_ready else "connecting"})


@app.route("/api/labels", methods=["GET"])
def api_labels() -> Response:
    return jsonify({"labels": _available_labels()})


@app.route("/api/capture-label", methods=["POST"])
def api_capture_label():
    global _capture_labels
    if _recognizer is None:
        return jsonify({"error": "Object recognition unavailable"}), 503
    data = request.get_json(silent=True) or {}
    label = str(data.get("label", "")).strip()
    if not label:
        _capture_labels = []
        return jsonify({"labels": _capture_labels})
    if label not in _available_labels():
        return jsonify({"error": "Unknown label"}), 400
    _capture_labels = [label]
    return jsonify({"labels": _capture_labels})


@app.route("/api/capture-settings", methods=["GET", "POST"])
def api_capture_settings() -> Response:
    global _shoot_mode, _burst_count
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        mode = data.get("mode")
        burst_count = data.get("burst_count")

        if mode is not None:
            normalized_mode = str(mode).strip().lower()
            if normalized_mode not in (_SHOOT_MODE_SINGLE, _SHOOT_MODE_BURST):
                return jsonify({"error": "Invalid mode"}), 400
            _shoot_mode = normalized_mode

        if burst_count is not None:
            try:
                parsed_count = int(burst_count)
            except (TypeError, ValueError):
                return jsonify({"error": "Invalid burst_count"}), 400
            if not (_MIN_BURST_COUNT <= parsed_count <= _MAX_BURST_COUNT):
                return (
                    jsonify(
                        {
                            "error": f"burst_count must be between {_MIN_BURST_COUNT} and {_MAX_BURST_COUNT}",
                        }
                    ),
                    400,
                )
            _burst_count = parsed_count

        if _shoot_mode == _SHOOT_MODE_BURST and _burst_count is None:
            _burst_count = _DEFAULT_BURST_COUNT

    return jsonify(_capture_settings_payload())


@app.route("/api/capture-mode", methods=["GET", "POST"])
def api_capture_mode() -> Response:
    global _auto_capture_enabled
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        requested_enabled = bool(data.get("enabled"))
        if requested_enabled and _active_ui_mode != _MODE_AUTO:
            return jsonify({"error": "Auto capture is only available in Auto mode"}), 400
        _auto_capture_enabled = requested_enabled and _active_ui_mode == _MODE_AUTO
    if _active_ui_mode != _MODE_AUTO and _auto_capture_enabled:
        _auto_capture_enabled = False
    return jsonify({"enabled": _auto_capture_enabled})


@app.route("/api/ui-mode", methods=["GET", "POST"])
def api_ui_mode() -> Response:
    global _active_ui_mode, _auto_capture_enabled
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        requested_mode = str(data.get("mode", "")).strip().lower()
        if requested_mode not in _VALID_UI_MODES:
            return jsonify({"error": "Invalid mode"}), 400
        _active_ui_mode = requested_mode
        if _active_ui_mode != _MODE_AUTO:
            _auto_capture_enabled = False
    return jsonify({"mode": _active_ui_mode})


@app.route("/api/manual-capture", methods=["POST"])
def api_manual_capture() -> Response:
    global _camera, _stream_proc, _stream_thread, _frame_queue
    if _camera is None:
        return jsonify({"error": "Camera unavailable"}), 503
    if _active_ui_mode != _MODE_MANUAL:
        return (
            jsonify({"error": "Manual capture is only available while Manual mode is active"}),
            409,
        )
    data = request.get_json(silent=True) or {}
    requested_mode = str(data.get("mode", _shoot_mode)).strip().lower()
    if requested_mode not in (_SHOOT_MODE_SINGLE, _SHOOT_MODE_BURST):
        return jsonify({"error": "Invalid mode"}), 400

    resolved_burst_count = _burst_count
    if requested_mode == _SHOOT_MODE_BURST:
        burst_value = data.get("burst_count", _burst_count)
        try:
            resolved_burst_count = int(burst_value)
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid burst_count"}), 400
        if not (_MIN_BURST_COUNT <= resolved_burst_count <= _MAX_BURST_COUNT):
            return (
                jsonify(
                    {
                        "error": f"burst_count must be between {_MIN_BURST_COUNT} and {_MAX_BURST_COUNT}",
                    }
                ),
                400,
            )

    capture_path = None
    error_message = None
    with _stream_lock:
        _camera._clear_frame_queue(_frame_queue)
        _camera._debug_log_queue("manual_capture", _frame_queue)
        _camera._stop_preview_process(_stream_proc, _stream_thread)
        _stream_proc = None
        _stream_thread = None
        _frame_queue = None
        try:
            if requested_mode == _SHOOT_MODE_SINGLE:
                capture_path = _camera.capture_single()
            else:
                capture_path = _camera.capture_burst(resolved_burst_count)
            print(f"Manual capture saved to {capture_path}")
        except Exception as exc:
            error_message = str(exc)
        finally:
            try:
                _stream_proc, _stream_thread, _frame_queue = _camera.start_live_preview_stream()
            except Exception as restart_error:
                print(f"Unable to restart live preview stream: {restart_error}")
                _stream_proc = None
                _stream_thread = None
                _frame_queue = None

    if error_message is not None:
        return jsonify({"error": error_message}), 500

    return jsonify(
        {
            "status": "ok",
            "mode": requested_mode,
            "burst_count": resolved_burst_count if requested_mode == _SHOOT_MODE_BURST else None,
            "path": str(capture_path) if capture_path is not None else None,
        }
    )


atexit.register(_shutdown_stream)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True, debug=True)
