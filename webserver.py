import atexit
import io
import queue
import re
import threading
import time
import webbrowser
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Generator, List, Optional

import cv2
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
)
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

_TIMELAPSE_MIN_INTERVAL_SECONDS = 5.0
_TIMELAPSE_MAX_INTERVAL_SECONDS = 3600.0  # 1 hour between frames
_TIMELAPSE_MIN_DURATION_SECONDS = 60.0
_TIMELAPSE_MAX_DURATION_SECONDS = 21600.0  # 6 hours total runtime
_TIMELAPSE_DEFAULT_INTERVAL_SECONDS = 30.0
_TIMELAPSE_DEFAULT_DURATION_SECONDS = 600.0
_TIMELAPSE_DIRECTORY_PREFIX = "timelapse_"

_timelapse_thread: Optional[threading.Thread] = None
_timelapse_stop_event: Optional[threading.Event] = None
_timelapse_status: Dict[str, object] = {
    "active": False,
    "interval_seconds": _TIMELAPSE_DEFAULT_INTERVAL_SECONDS,
    "duration_seconds": _TIMELAPSE_DEFAULT_DURATION_SECONDS,
    "captures_completed": 0,
    "planned_captures": 0,
    "started_at": None,
    "ended_at": None,
    "last_error": None,
    "capture_directory": None,
}

_timelapse_capture_dir: Optional[Path] = None

_BASE_DIR = Path().home() / "Documents" / "Camera Control"
_BASE_DIR.mkdir(parents=True, exist_ok=True)
_PREVIEW_ROOT = (_BASE_DIR / "captures/previews").resolve()
_PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)
_RAW_ROOT = (_BASE_DIR / "captures/raw").resolve()
_RAW_ROOT.mkdir(parents=True, exist_ok=True)
_PREVIEW_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_RAW_EXTENSIONS = {".nef"}
_TIMESTAMP_PATTERN = re.compile(r"(\d{8}-\d{6})")
_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
_GROUPED_CAPTURE_PREFIXES = {
    "burst-": "burst",
    _TIMELAPSE_DIRECTORY_PREFIX: "timelapse",
}


def _schedule_browser_launch(url: str, delay_seconds: float = 1.0) -> None:
    """Open the default browser after a small delay once the server starts."""

    def _open() -> None:
        try:
            webbrowser.open(url)
        except Exception as exc:  # noqa: BLE001 - just log failures
            print(f"Unable to open browser automatically: {exc}")

    threading.Timer(delay_seconds, _open).start()


def _parse_capture_timestamp(value: str) -> Optional[datetime]:
    if not value:
        return None
    match = _TIMESTAMP_PATTERN.search(value)
    if not match:
        return None
    raw_value = match.group(1)
    try:
        return datetime.strptime(raw_value, _TIMESTAMP_FORMAT)
    except ValueError:
        return None


def _format_capture_timestamp(value: datetime) -> str:
    return value.strftime("%b %d, %Y \n %H:%M:%S")


def _assign_day_metadata(items: List[Dict[str, object]]) -> None:
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    for item in items:
        timestamp_iso = item.get("timestamp_iso")
        if not timestamp_iso:
            continue
        try:
            timestamp_value = datetime.fromisoformat(timestamp_iso)
        except (TypeError, ValueError):
            continue
        capture_date = timestamp_value.date()
        if capture_date == today:
            day_label = "Today"
        elif capture_date == yesterday:
            day_label = "Yesterday"
        else:
            day_label = capture_date.strftime("%B %d, %Y")
        item["day_key"] = capture_date.isoformat()
        item["day_label"] = day_label


def _relative_preview_path(path: Path) -> Optional[str]:
    try:
        return path.relative_to(_PREVIEW_ROOT).as_posix()
    except ValueError:
        return None


def _is_preview_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _PREVIEW_EXTENSIONS


def _relative_raw_path_from_preview(
    relative_preview_path: Optional[str],
) -> Optional[str]:
    if not relative_preview_path:
        return None
    normalized = Path(relative_preview_path)
    if normalized.is_absolute() or ".." in normalized.parts:
        return None
    raw_candidate = (_RAW_ROOT / normalized).with_suffix(".nef")
    try:
        raw_relative = raw_candidate.relative_to(_RAW_ROOT)
    except ValueError:
        return None
    if not raw_candidate.exists():
        return None
    return raw_relative.as_posix()


def _resolve_capture_directory(root: Path, capture_id: str) -> Optional[Path]:
    """Return a validated directory inside the given root for stack downloads."""
    if not capture_id:
        return None
    normalized = Path(capture_id)
    if normalized.is_absolute() or ".." in normalized.parts:
        return None
    target_dir = (root / normalized).resolve()
    try:
        target_dir.relative_to(root)
    except ValueError:
        return None
    if not target_dir.exists() or not target_dir.is_dir():
        return None
    return target_dir


def _zip_capture_directory(
    source_dir: Path, allowed_extensions: Optional[set] = None
) -> Optional[io.BytesIO]:
    archive_stream = io.BytesIO()
    file_count = 0
    with zipfile.ZipFile(archive_stream, "w", zipfile.ZIP_DEFLATED) as bundle:
        for file_path in sorted(source_dir.rglob("*")):
            if file_path.is_dir():
                continue
            if (
                allowed_extensions
                and file_path.suffix.lower() not in allowed_extensions
            ):
                continue
            arcname = file_path.relative_to(source_dir).as_posix()
            bundle.write(file_path, arcname=arcname)
            file_count += 1
    if file_count == 0:
        return None
    archive_stream.seek(0)
    return archive_stream


def _build_single_capture_item(file_path: Path) -> Optional[Dict[str, object]]:
    if not _is_preview_image(file_path):
        return None
    timestamp = _parse_capture_timestamp(file_path.stem)
    if timestamp is None:
        return None
    relative_path = _relative_preview_path(file_path)
    if relative_path is None:
        return None
    sort_key = timestamp.timestamp()
    raw_path = _relative_raw_path_from_preview(relative_path)
    image_payload = {
        "path": relative_path,
        "filename": file_path.name,
        "raw_path": raw_path,
    }
    return {
        "id": file_path.stem,
        "type": "single",
        "count": 1,
        "timestamp_label": _format_capture_timestamp(timestamp),
        "timestamp_iso": timestamp.isoformat(),
        "sort_key": sort_key,
        "cover_path": relative_path,
        "images": [image_payload],
    }


def _build_grouped_capture_item(directory: Path) -> Optional[Dict[str, object]]:
    if not directory.is_dir():
        return None
    capture_type = None
    lower_name = directory.name.lower()
    for prefix, type_label in _GROUPED_CAPTURE_PREFIXES.items():
        if lower_name.startswith(prefix):
            capture_type = type_label
            break
    if capture_type is None:
        return None
    timestamp = _parse_capture_timestamp(directory.name)
    if timestamp is None:
        return None
    child_images = []
    for child in sorted(directory.iterdir()):
        if not _is_preview_image(child):
            continue
        relative_path = _relative_preview_path(child)
        if relative_path is None:
            continue
        raw_path = _relative_raw_path_from_preview(relative_path)
        child_images.append(
            {
                "path": relative_path,
                "filename": child.name,
                "raw_path": raw_path,
            }
        )
    if not child_images:
        return None
    sort_key = timestamp.timestamp()
    return {
        "id": directory.name,
        "type": capture_type,
        "count": len(child_images),
        "timestamp_label": _format_capture_timestamp(timestamp),
        "timestamp_iso": timestamp.isoformat(),
        "sort_key": sort_key,
        "cover_path": child_images[0]["path"],
        "images": child_images,
        "stack_id": f"stack-{directory.name}",
    }


def _gather_gallery_items() -> List[Dict[str, object]]:
    if not _PREVIEW_ROOT.exists():
        return []
    try:
        entries = list(_PREVIEW_ROOT.iterdir())
    except FileNotFoundError:
        return []
    items: List[Dict[str, object]] = []
    for entry in entries:
        item: Optional[Dict[str, object]] = None
        if entry.is_dir():
            item = _build_grouped_capture_item(entry)
        else:
            item = _build_single_capture_item(entry)
        if item is not None:
            items.append(item)
    items.sort(key=lambda data: data.get("sort_key", 0.0), reverse=True)
    for item in items:
        item.pop("sort_key", None)
    _assign_day_metadata(items)
    return items


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


def _planned_timelapse_captures(
    interval_seconds: float, duration_seconds: float
) -> int:
    if interval_seconds <= 0 or duration_seconds <= 0:
        return 0
    planned = int(duration_seconds // interval_seconds) + 1
    return max(1, planned)


def _timelapse_state_payload() -> Dict[str, object]:
    payload = dict(_timelapse_status)
    for timestamp_key in ("started_at", "ended_at"):
        value = payload.get(timestamp_key)
        if isinstance(value, datetime):
            payload[timestamp_key] = value.isoformat()
    payload.update(
        {
            "min_interval_seconds": _TIMELAPSE_MIN_INTERVAL_SECONDS,
            "max_interval_seconds": _TIMELAPSE_MAX_INTERVAL_SECONDS,
            "min_duration_seconds": _TIMELAPSE_MIN_DURATION_SECONDS,
            "max_duration_seconds": _TIMELAPSE_MAX_DURATION_SECONDS,
        }
    )
    return payload


def _perform_timelapse_capture() -> Optional[str]:
    global _camera, _stream_proc, _stream_thread, _frame_queue, _timelapse_capture_dir
    if _camera is None:
        raise RuntimeError("Camera unavailable")
    capture_path = None
    with _stream_lock:
        _camera._clear_frame_queue(_frame_queue)
        _camera._debug_log_queue("timelapse", _frame_queue)
        _camera._stop_preview_process(_stream_proc, _stream_thread)
        _stream_proc = None
        _stream_thread = None
        _frame_queue = None
        try:
            capture_path = _camera.capture_single(
                destination_dir=_timelapse_capture_dir
            )
            print(f"Timelapse capture saved to {capture_path}")
        finally:
            try:
                _stream_proc, _stream_thread, _frame_queue = (
                    _camera.start_live_preview_stream()
                )
            except Exception as restart_error:
                print(
                    f"Unable to restart live preview stream after timelapse capture: {restart_error}"
                )
                _stream_proc = None
                _stream_thread = None
                _frame_queue = None
    return str(capture_path) if capture_path is not None else None


def _timelapse_worker(
    interval_seconds: float, duration_seconds: float, stop_event: threading.Event
) -> None:
    global _timelapse_thread, _timelapse_stop_event, _timelapse_capture_dir
    start_time = time.monotonic()
    next_capture_time = start_time
    while not stop_event.is_set():
        now = time.monotonic()
        wait_time = next_capture_time - now
        if wait_time > 0:
            stop_event.wait(wait_time)
            continue
        try:
            _perform_timelapse_capture()
            _timelapse_status["captures_completed"] = (
                int(_timelapse_status.get("captures_completed", 0)) + 1
            )
        except Exception as exc:  # noqa: BLE001 - surface capture issues in logs
            _timelapse_status["last_error"] = str(exc)
            print(f"Timelapse capture failed: {exc}")
            break
        if stop_event.is_set():
            break
        next_capture_time += interval_seconds
        if next_capture_time - start_time > duration_seconds:
            break

    _timelapse_status["active"] = False
    _timelapse_status["ended_at"] = datetime.utcnow()
    _timelapse_capture_dir = None
    _timelapse_status["capture_directory"] = None
    _timelapse_thread = None
    _timelapse_stop_event = None


def _start_timelapse(interval_seconds: float, duration_seconds: float) -> None:
    global _timelapse_thread, _timelapse_stop_event, _timelapse_capture_dir
    resolved_interval = float(interval_seconds)
    resolved_duration = float(duration_seconds)
    timestamp_label = datetime.utcnow().strftime(_TIMESTAMP_FORMAT)
    base_capture_dir = getattr(_camera, "capture_dir", Path("captures/raw"))
    timelapse_dir = (
        Path(base_capture_dir) / f"{_TIMELAPSE_DIRECTORY_PREFIX}{timestamp_label}"
    )
    timelapse_dir.mkdir(parents=True, exist_ok=True)
    _timelapse_capture_dir = timelapse_dir
    stop_event = threading.Event()
    worker = threading.Thread(
        target=_timelapse_worker,
        args=(resolved_interval, resolved_duration, stop_event),
        daemon=True,
    )
    _timelapse_stop_event = stop_event
    _timelapse_thread = worker
    _timelapse_status.update(
        {
            "active": True,
            "interval_seconds": resolved_interval,
            "duration_seconds": resolved_duration,
            "captures_completed": 0,
            "planned_captures": _planned_timelapse_captures(
                resolved_interval, resolved_duration
            ),
            "started_at": datetime.utcnow(),
            "ended_at": None,
            "last_error": None,
            "capture_directory": timelapse_dir.as_posix(),
        }
    )
    worker.start()


def _stop_timelapse(force: bool = False) -> bool:
    global _timelapse_thread, _timelapse_stop_event, _timelapse_capture_dir
    stop_event = _timelapse_stop_event
    worker = _timelapse_thread
    if stop_event is None or worker is None:
        return False
    stop_event.set()
    timeout = 5.0 if not force else 0.0
    worker.join(timeout=timeout)
    still_running = worker.is_alive()
    if still_running:
        print("Timelapse worker did not exit before timeout")
    _timelapse_thread = None
    _timelapse_stop_event = None
    _timelapse_status["active"] = False
    if _timelapse_status.get("ended_at") is None:
        _timelapse_status["ended_at"] = datetime.utcnow()
    if not still_running:
        _timelapse_capture_dir = None
        _timelapse_status["capture_directory"] = None
    return not still_running


def _shutdown_timelapse() -> None:
    try:
        _stop_timelapse(force=True)
    except Exception as exc:  # noqa: BLE001 - avoid noisy shutdowns
        print(f"Error while stopping timelapse thread: {exc}")


def _shutdown_stream() -> None:
    global _stream_proc, _frame_queue, _camera
    proc = _stream_proc
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()
    _stream_proc = None
    _frame_queue = None
    if _camera is not None:
        try:
            _camera._set_state("disconnected")
        except Exception:
            pass


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
    latest_capture = next(iter(_gather_gallery_items()), None)
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
        timelapse_url=url_for("api_timelapse"),
        latest_capture=latest_capture,
    )


@app.route("/gallery")
def gallery() -> str:
    gallery_items = _gather_gallery_items()
    total_frames = sum(item.get("count", 0) for item in gallery_items)
    return render_template(
        "gallery.html",
        gallery_items=gallery_items,
        total_frames=total_frames,
        capture_sets=len(gallery_items),
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


@app.route("/previews/<path:filename>")
def serve_preview(filename: str) -> Response:
    normalized = Path(filename)
    if normalized.is_absolute() or ".." in normalized.parts:
        abort(404)
    if not _PREVIEW_ROOT.exists():
        abort(404)
    return send_from_directory(str(_PREVIEW_ROOT), normalized.as_posix())


@app.route("/raw/<path:filename>")
def serve_raw(filename: str) -> Response:
    normalized = Path(filename)
    if normalized.is_absolute() or ".." in normalized.parts:
        abort(404)
    if not _RAW_ROOT.exists():
        abort(404)
    target_path = _RAW_ROOT / normalized
    if not target_path.exists():
        abort(404)
    return send_from_directory(
        str(_RAW_ROOT), normalized.as_posix(), as_attachment=True
    )


@app.route("/download-stack/<capture_id>/<kind>")
def download_stack(capture_id: str, kind: str) -> Response:
    kind_lower = (kind or "").strip().lower()
    if kind_lower not in {"preview", "raw"}:
        abort(404)

    root = _PREVIEW_ROOT if kind_lower == "preview" else _RAW_ROOT
    allowed_extensions = (
        _PREVIEW_EXTENSIONS if kind_lower == "preview" else _RAW_EXTENSIONS
    )
    target_dir = _resolve_capture_directory(root, capture_id)
    if target_dir is None:
        abort(404)

    archive_stream = _zip_capture_directory(
        target_dir, allowed_extensions=allowed_extensions
    )
    if archive_stream is None:
        abort(404)

    download_name = f"{capture_id}-{kind_lower}.zip"
    return send_file(
        archive_stream,
        mimetype="application/zip",
        as_attachment=True,
        download_name=download_name,
    )


@app.route("/api/health")
def health():
    stream_ready = _stream_proc is not None and _stream_proc.poll() is None
    camera_state = getattr(_camera, "state", "disconnected")
    return jsonify(
        {
            "status": (
                "ok"
                if stream_ready and camera_state == "ready"
                else "capturing" if camera_state == "capturing" else "connecting"
            ),
        }
    )


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
            return (
                jsonify({"error": "Auto capture is only available in Auto mode"}),
                400,
            )
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
        if _active_ui_mode != _MODE_TIMELAPSE and _timelapse_status.get("active"):
            _stop_timelapse()
    return jsonify({"mode": _active_ui_mode})


@app.route("/api/manual-capture", methods=["POST"])
def api_manual_capture() -> Response:
    global _camera, _stream_proc, _stream_thread, _frame_queue
    if _camera is None:
        return jsonify({"error": "Camera unavailable"}), 503
    if _timelapse_status.get("active"):
        return jsonify({"error": "Timelapse is currently running"}), 409
    if _active_ui_mode != _MODE_MANUAL:
        return (
            jsonify(
                {
                    "error": "Manual capture is only available while Manual mode is active"
                }
            ),
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
                _stream_proc, _stream_thread, _frame_queue = (
                    _camera.start_live_preview_stream()
                )
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
            "burst_count": (
                resolved_burst_count if requested_mode == _SHOOT_MODE_BURST else None
            ),
            "path": str(capture_path) if capture_path is not None else None,
        }
    )


@app.route("/api/timelapse", methods=["GET", "POST"])
def api_timelapse() -> Response:
    if request.method == "GET":
        return jsonify(_timelapse_state_payload())

    data = request.get_json(silent=True) or {}
    action = str(data.get("action", "")).strip().lower()
    if action not in {"start", "stop"}:
        return jsonify({"error": "Unknown timelapse action"}), 400

    if action == "stop":
        _stop_timelapse()
        return jsonify(_timelapse_state_payload())

    if _camera is None:
        return jsonify({"error": "Camera unavailable"}), 503
    if _active_ui_mode != _MODE_TIMELAPSE:
        return (
            jsonify(
                {"error": "Timelapse can only start while Timelapse mode is active"}
            ),
            409,
        )
    if _timelapse_status.get("active"):
        return jsonify({"error": "Timelapse already running"}), 409

    try:
        interval_value = int(
            float(data.get("interval_seconds", _timelapse_status["interval_seconds"]))
        )
        duration_value = int(
            float(data.get("duration_seconds", _timelapse_status["duration_seconds"]))
        )
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid interval or duration"}), 400

    if not (
        _TIMELAPSE_MIN_INTERVAL_SECONDS
        <= interval_value
        <= _TIMELAPSE_MAX_INTERVAL_SECONDS
    ):
        return (
            jsonify(
                {
                    "error": f"interval_seconds must be between {_TIMELAPSE_MIN_INTERVAL_SECONDS:.0f} and {_TIMELAPSE_MAX_INTERVAL_SECONDS:.0f}",
                }
            ),
            400,
        )
    if not (
        _TIMELAPSE_MIN_DURATION_SECONDS
        <= duration_value
        <= _TIMELAPSE_MAX_DURATION_SECONDS
    ):
        return (
            jsonify(
                {
                    "error": f"duration_seconds must be between {_TIMELAPSE_MIN_DURATION_SECONDS:.0f} and {_TIMELAPSE_MAX_DURATION_SECONDS:.0f}",
                }
            ),
            400,
        )

    _start_timelapse(float(interval_value), float(duration_value))
    return jsonify(_timelapse_state_payload())


atexit.register(_shutdown_stream)
atexit.register(_shutdown_timelapse)

if __name__ == "__main__":
    _schedule_browser_launch("http://127.0.0.1:8000/", delay_seconds=1.5)
    app.run(host="0.0.0.0", port=8000, threaded=True, debug=True)
