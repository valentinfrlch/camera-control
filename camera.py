import subprocess
import cv2
import numpy as np
import threading
import queue
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING
import rawpy

if TYPE_CHECKING:
    from object_recognition import ObjectRecognizer


class Camera:
    def __init__(self, auto_release=True, capture_confidence_threshold: float = 0.6):
        self.auto_release = auto_release
        self.capture_dir = Path("captures/raw")
        self.capture_cooldown_seconds = 3.0
        self.capture_confidence_threshold = capture_confidence_threshold
        self._last_capture_timestamp = 0.0
        self._frame_queue: Optional["queue.Queue"] = None
        if self.auto_release:
            self._release_ptpcamera()

    def _release_ptpcamera(self):
        """Best effort stop of macOS PTPCamera so gphoto2 can claim USB."""
        if sys.platform != "darwin":
            return
        try:
            subprocess.run(
                ["killall", "ptpcamerad"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            pass

    def live_preview(self, recognizer: Optional["ObjectRecognizer"] = None):
        """
        Stream frames to an OpenCV window and optionally overlay object recognition results.
        """
        proc: Optional[subprocess.Popen] = None
        reader_thread: Optional[threading.Thread] = None
        window = "Live Preview"
        window_created = False
        try:
            proc, reader_thread, q = self.start_live_preview_stream()
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            window_created = True
            active_recognizer = recognizer

            while True:
                try:
                    frame = q.get(timeout=3)
                except queue.Empty:
                    if proc.poll() is not None:
                        break
                    continue

                display_frame = frame
                capture_requested = False
                if active_recognizer is not None:
                    try:
                        display_frame, detections = active_recognizer.annotate(frame)
                        capture_requested = self._should_trigger_capture(
                            getattr(active_recognizer, "capture_objects", []),
                            detections,
                        )
                    except Exception as detection_error:
                        print(f"Object recognition disabled: {detection_error}")
                        active_recognizer = None
                        capture_requested = False

                cv2.imshow(window, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

                if capture_requested:
                    self._clear_frame_queue(self._frame_queue)
                    self._debug_log_queue("live_preview", self._frame_queue)
                    proc, reader_thread, q = self._restart_preview_with_capture(
                        proc, reader_thread, self._frame_queue
                    )

        except Exception as e:
            print(f"Error: {e}")
        finally:
            if window_created:
                try:
                    cv2.destroyWindow(window)
                except cv2.error:
                    pass
            self._stop_preview_process(proc, reader_thread)

    def capture_single(self, destination_dir: Optional[Path] = None) -> Path:
        """Capture a single image, optionally forcing the destination directory."""
        destination = self._default_capture_path(destination_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "gphoto2",
            "--capture-image-and-download",
            f"--filename={str(destination)}",
            "--debug-loglevel=error",
        ]

        try:
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except FileNotFoundError as err:
            raise RuntimeError("gphoto2 is not installed or not found in PATH") from err
        except subprocess.CalledProcessError as err:
            raise RuntimeError(f"gphoto2 failed to capture image: {err}") from err

        if not destination.exists():
            raise RuntimeError("gphoto2 reported success but no file was created")
        # convert to jpeg
        self._convert_to_jpeg(destination)
        return destination

    def capture_burst(self, burst_count: int = 3) -> Path:
        """Capture multiple images in burst mode and store them in a unique folder."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        burst_dir = self.capture_dir / f"burst-{timestamp}"
        burst_dir.mkdir(parents=True, exist_ok=True)

        # Use gphoto2's %n placeholder so every frame in the burst keeps a unique filename.
        filename_template = burst_dir / "capture-%03n.nef"

        cmd = [
            "gphoto2",
            "--set-config",
            "capturemode=Burst",
            "--set-config",
            f"burstnumber={burst_count}",
            "--capture-image-and-download",
            "--filename",
            str(filename_template),
            "--debug-loglevel=error",
        ]

        try:
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except FileNotFoundError as err:
            raise RuntimeError("gphoto2 is not installed or not found in PATH") from err
        except subprocess.CalledProcessError as err:
            raise RuntimeError(f"gphoto2 failed to capture image: {err}") from err

        captured_files = sorted(burst_dir.glob("*.nef"))
        if not captured_files:
            raise RuntimeError(
                "gphoto2 reported success but no burst files were created"
            )

        # convert all to jpeg, ensure burst directory for previews exists and jpgs go there
        for nef_path in captured_files:
            self._convert_to_jpeg(nef_path)

        return burst_dir

    def _default_capture_path(self, parent_dir: Optional[Path] = None) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target_dir = parent_dir or self.capture_dir
        return target_dir / f"capture-{timestamp}.nef"

    @staticmethod
    def _convert_to_jpeg(nef_path: Path) -> Path:
        """Convert a NEF stored under captures/raw into a mirrored previews JPG."""
        with rawpy.imread(str(nef_path)) as raw:
            rgb = raw.postprocess(rawpy.Params(use_camera_wb=True))  # type: ignore

            # downscale to max 1920 width for previews
            height, width = rgb.shape[:2]
            if width > 1920:
                scale = 1920 / width
                rgb = cv2.resize(
                    rgb,
                    (int(width * scale), int(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )

        raw_root = Path("captures/raw")
        preview_root = Path("captures/previews")
        try:
            relative_path = nef_path.relative_to(raw_root)
            jpeg_path = preview_root / relative_path
        except ValueError:
            # fall back to placing JPEG alongside the source if it wasn't under /raw
            jpeg_path = nef_path
        jpeg_path = jpeg_path.with_suffix(".jpg")
        jpeg_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(jpeg_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return jpeg_path

    def start_live_preview_stream(self, chunk_size=16384):
        """
        Start background thread that reads stdout MJPEG and enqueues decoded frames.
        Returns (proc, thread, frame_queue).
        """
        cmd = ["gphoto2", "--stdout", "--capture-movie", "--debug-loglevel=error"]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
        )
        frame_queue = queue.Queue(maxsize=1)  # small buffer
        self._frame_queue = frame_queue
        SOI, EOI = b"\xff\xd8", b"\xff\xd9"

        def reader():
            buf = bytearray()
            try:
                while True:
                    chunk = proc.stdout.read(chunk_size)  # type: ignore
                    if not chunk:
                        break
                    buf += chunk
                    while True:
                        start = buf.find(SOI)
                        if start == -1:
                            buf[:] = buf[-1:]
                            break
                        end = buf.find(EOI, start + 2)
                        if end == -1:
                            if start > 0:
                                del buf[:start]
                            break
                        frame_bytes = bytes(buf[start : end + 2])
                        del buf[: end + 2]
                        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            try:
                                frame_queue.put(frame, timeout=0.1)
                            except queue.Full:
                                pass
            finally:
                if proc.poll() is None:
                    proc.terminate()

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        return proc, t, frame_queue

    def _should_trigger_capture(
        self,
        capture_labels: Optional[List[str]],
        detections: Optional[List[Dict[str, object]]],
    ) -> bool:
        if not capture_labels or not detections:
            return False
        now = time.monotonic()
        if now - self._last_capture_timestamp < self.capture_cooldown_seconds:
            return False
        target_labels = {label.lower() for label in capture_labels}
        for detection in detections:
            detected_label = str(detection.get("label", "")).lower()
            confidence = float(detection.get("confidence", 0.0))  # type: ignore
            if (
                detected_label in target_labels
                and confidence >= self.capture_confidence_threshold
            ):
                print(
                    f"Triggering capture for detected object: {detected_label} ({confidence:.2%})"
                )
                self._last_capture_timestamp = now
                return True
        return False

    def _restart_preview_with_capture(
        self,
        proc: Optional[subprocess.Popen],
        reader_thread: Optional[threading.Thread],
        frame_queue: Optional["queue.Queue"] = None,
        shoot_mode: str = "burst",
        burst_count: int = 3,
    ):
        self._stop_preview_process(proc, reader_thread)
        target_queue = frame_queue or self._frame_queue
        self._clear_frame_queue(target_queue)
        self._debug_log_queue("restart_preview", target_queue)
        try:
            if str(shoot_mode).lower() == "single":
                capture_path = self.capture_single()
            else:
                capture_path = self.capture_burst(burst_count)
            print(f"Captured image saved to {capture_path}")
        except RuntimeError as capture_error:
            print(f"Capture failed: {capture_error}")
        return self.start_live_preview_stream()

    @staticmethod
    def _clear_frame_queue(frame_queue: Optional["queue.Queue"]) -> None:
        if frame_queue is None:
            return
        try:
            while True:
                frame_queue.get_nowait()
        except queue.Empty:
            pass

    def _debug_log_queue(
        self, context: str, frame_queue: Optional["queue.Queue"]
    ) -> None:
        if frame_queue is None:
            print(f"[debug] {context}: frame queue=None")
            return
        internal = getattr(frame_queue, "queue", None)
        print(f"[debug] {context}: size={frame_queue.qsize()} internal={internal}")

    def _stop_preview_process(
        self,
        proc: Optional[subprocess.Popen],
        reader_thread: Optional[threading.Thread],
    ) -> None:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
        if reader_thread is not None and reader_thread.is_alive():
            reader_thread.join(timeout=2)
