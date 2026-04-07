"""
auto_track.py — Autonomous visual tracking for the DJI Tello.

Uses YOLOv8 to detect a person in the drone's video feed
and a PID controller to keep the target centered via continuous yaw, altitude,
and forward/backward adjustments.

Usage:
  uv run python auto_track.py
  uv run python auto_track.py --no-fly   # debug without takeoff
  uv run python auto_track.py --debug    # enable detailed tracking logs
"""

import argparse
import logging
import os
os.environ["OPENH264_LIBRARY"] = "NUL"
import signal
import sys
import time
from datetime import datetime
import cv2
import numpy as np

from tello_controller import DroneManager, DroneState

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("tello.tracker")

# ── PID Controller ───────────────────────────────────────────────

YAW_PID_GAINS = (0.15, 0.0, 0.02)
ALT_PID_GAINS = (0.15, 0.0, 0.02)
FB_PID_GAINS  = (0.0009, 0.0, 0.0001)
TARGET_AREA   = 45_000   # px²; tuned for close in-flight standoff distance
MAX_FB_SPEED  = 45       # cap forward/back RC at ±45 to prevent aggressive oscillations

YAW_DEADZONE  = 15       # px — ignore horizontal jitter below this
ALT_DEADZONE  = 15       # px — ignore vertical jitter below this
AREA_DEADZONE = 2000     # px² — ignore area jitter below this

OUTPUT_DIR = "recordings"

INTEGRAL_LIMIT = 400.0
VIDEO_TIMEOUT_SEC = 5.0
MAX_CONSECUTIVE_ERRORS = 30


class PIDController:
    """Simple time-based PID controller with output clamping and anti-windup."""

    def __init__(
        self,
        kP: float,
        kI: float,
        kD: float,
        output_limits: tuple[float, float] = (-100, 100),
    ) -> None:
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self._limits = output_limits
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = time.time()

    def compute(self, error: float) -> float:
        now = time.time()
        dt = max(now - self._prev_time, 1e-6)

        self._integral += error * dt
        self._integral = max(-INTEGRAL_LIMIT, min(INTEGRAL_LIMIT, self._integral))
        derivative = (error - self._prev_error) / dt

        output = self.kP * error + self.kI * self._integral + self.kD * derivative

        self._prev_error = error
        self._prev_time = now

        return max(self._limits[0], min(self._limits[1], output))

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = time.time()


# ── Drone Tracker ────────────────────────────────────────────────

class DroneTracker:
    """Autonomous visual tracker that keeps a detected target centered."""

    def __init__(self, drone: DroneManager) -> None:
        self._drone = drone
        self._yaw_pid = PIDController(*YAW_PID_GAINS, output_limits=(-60, 60))
        self._alt_pid = PIDController(*ALT_PID_GAINS, output_limits=(-60, 60))
        self._fb_pid  = PIDController(*FB_PID_GAINS, output_limits=(-MAX_FB_SPEED, MAX_FB_SPEED))
        self._video_writer: cv2.VideoWriter | None = None
        self._recording = False
        self._rec_path: str = ""
        self._last_target_center: tuple[int, int] | None = None
        self._yolo = self._load_yolo()

    @staticmethod
    def _load_yolo():
        from ultralytics import YOLO
        # Load base model, export to Intel format, and return the fast version
        model = YOLO("yolov8n.pt")
        model.export(format="openvino", imgsz=640)
        return YOLO("yolov8n_openvino_model/")

    def _detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return list of (x, y, w, h) bounding boxes."""
        
        # Add imgsz=640 back into this line to keep the math fast!
        results = self._yolo(frame, verbose=False, imgsz=640)[0] 
        
        boxes = []
        for r in results.boxes:
            if int(r.cls[0]) != 0:  # filter to person class only
                continue
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes

    def _select_target(
        self,
        detections: list[tuple[int, int, int, int]],
    ) -> tuple[int, int, int, int] | None:
        """Spatial proximity target lock — stay on the same person across frames."""
        if not detections:
            return None

        if self._last_target_center is None:
            # First lock — pick the largest person
            target = max(detections, key=lambda b: b[2] * b[3])
        else:
            # Pick the detection closest to the last known target position
            lx, ly = self._last_target_center
            target = min(detections, key=lambda b: (b[0] + b[2] // 2 - lx) ** 2 + (b[1] + b[3] // 2 - ly) ** 2)

        self._last_target_center = (target[0] + target[2] // 2, target[1] + target[3] // 2)
        return target

    def _toggle_recording(self, fps: float) -> None:
        if self._recording:
            self._stop_recording()
        else:
            self._rec_path = os.path.join(OUTPUT_DIR, f"rec_{datetime.now():%Y%m%d_%H%M%S}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            
            # Force exactly 15.0 FPS
            self._video_writer = cv2.VideoWriter(self._rec_path, fourcc, 15.0, (960, 720))
            if not self._video_writer.isOpened():
                logger.info("H.264 (avc1) unavailable — falling back to mp4v")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._video_writer = cv2.VideoWriter(self._rec_path, fourcc, 15.0, (960, 720))
            
            self._recording = True
            # Set a baseline timestamp for the frame timing
            self._next_frame_time = time.time()
            logger.info("Recording started at an enforced 15 FPS: %s", self._rec_path)

    def _stop_recording(self) -> None:
        if self._recording and self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            self._recording = False
            logger.info("Recording saved: %s", self._rec_path)

    def run(self) -> None:
        """Main tracking loop.

        Keys:
          q — quit (graceful land handled by caller)
          e — emergency motor kill
          l — graceful land
        """
        print("Tracker started. Keys: q=quit, l=land, e=EMERGENCY, r=record, s=screenshot")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        fps_timer = time.time()
        fps = 0
        frame_count = 0
        last_frame_time = time.time()
        consecutive_errors = 0
        take_screenshot = False

        while True:
            # ── Input handling ───────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                logger.info("Quit requested")
                break
            if key == ord("e"):
                logger.warning("EMERGENCY STOP — killing motors")
                self._drone.emergency_stop()
                break
            if key == ord("l"):
                logger.info("Graceful land requested")
                if self._drone.state == DroneState.FLYING:
                    self._drone.land()
                break
            if key == ord("s"):
                take_screenshot = True
            if key == ord("r"):
                self._toggle_recording(fps)

            # ── Frame acquisition ────────────────────────────────
            frame = self._drone.get_latest_frame()
            if frame is None:
                self._drone.send_rc(0, 0, 0, 0)
                if (
                    self._drone.state == DroneState.FLYING
                    and time.time() - last_frame_time > VIDEO_TIMEOUT_SEC
                ):
                    logger.critical(
                        "No video frame for %.0fs — auto-landing for safety",
                        VIDEO_TIMEOUT_SEC,
                    )
                    self._drone.land()
                    break
                time.sleep(0.033)
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            last_frame_time = time.time()

            # ── Detection & tracking ─────────────────────────────
            yaw_cmd = 0
            ud_cmd = 0
            fb_cmd = 0
            detections: list[tuple[int, int, int, int]] = []
            target = None
            try:
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2

                detections = self._detect(frame)
                target = self._select_target(detections)

                if target is not None:
                    tx, ty, tw, th = target
                    obj_cx = tx + tw // 2
                    obj_cy = ty + th // 2

                    x_err = obj_cx - cx
                    y_err = cy - obj_cy
                    area_err = (tw * th) - TARGET_AREA

                    # Apply deadzones — suppress PID response to small jitter
                    if abs(x_err) < YAW_DEADZONE:
                        x_err = 0
                    if abs(y_err) < ALT_DEADZONE:
                        y_err = 0
                    if abs(area_err) < AREA_DEADZONE:
                        area_err = 0

                    yaw_cmd = int(self._yaw_pid.compute(x_err))
                    ud_cmd = int(self._alt_pid.compute(y_err))
                    fb_cmd = -int(self._fb_pid.compute(area_err))

                    if frame_count == 0:  # log once per second
                        logger.debug(
                            "x_err=%+4d y_err=%+4d area=%d area_err=%+d → yaw=%+d ud=%+d fb=%+d",
                            obj_cx - cx, cy - obj_cy, tw * th, (tw * th) - TARGET_AREA,
                            yaw_cmd, ud_cmd, fb_cmd,
                        )
                else:
                    self._yaw_pid.reset()
                    self._alt_pid.reset()
                    self._fb_pid.reset()

                self._drone.send_rc(0, fb_cmd, ud_cmd, yaw_cmd)
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                logger.error("Tracking error (%d/%d): %s",
                             consecutive_errors, MAX_CONSECUTIVE_ERRORS, e)
                self._drone.send_rc(0, 0, 0, 0)
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.critical(
                        "Too many consecutive errors — auto-landing for safety"
                    )
                    if self._drone.state == DroneState.FLYING:
                        self._drone.land()
                    break
                continue

            # ── FPS counter ──────────────────────────────────────
            frame_count += 1
            if time.time() - fps_timer >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_timer = time.time()

            self._draw_debug(frame, detections, target, cx, cy, yaw_cmd, ud_cmd, fb_cmd, fps)
            cv2.imshow("Tello Tracker", frame)

            if take_screenshot:
                path = os.path.join(OUTPUT_DIR, f"shot_{datetime.now():%Y%m%d_%H%M%S}.png")
                cv2.imwrite(path, frame)
                logger.info("Screenshot saved: %s", path)
                take_screenshot = False

            if self._recording and self._video_writer is not None:
                now = time.time()
                # If YOLO lags, this loop writes the same frame multiple times to hit 15 FPS.
                # If YOLO spikes, it skips writing until 66ms have passed.
                while self._next_frame_time <= now:
                    self._video_writer.write(frame)
                    self._next_frame_time += (1.0 / 15.0)

        # Stop all RC movement and release resources
        self._stop_recording()
        self._drone.send_rc(0, 0, 0, 0)
        cv2.destroyAllWindows()

    def _draw_debug(
        self,
        frame: np.ndarray,
        detections: list[tuple[int, int, int, int]],
        target: tuple[int, int, int, int] | None,
        cx: int,
        cy: int,
        yaw: int,
        ud: int,
        fb: int,
        fps: int,
    ) -> None:
        """Draw bounding boxes, crosshairs, and HUD overlay."""
        battery = self._drone.battery

        # All detections in grey
        for x, y, w, h in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 180, 180), 1)

        # Target in green with line to center
        if target is not None:
            tx, ty, tw, th = target
            cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 2)
            obj_cx, obj_cy = tx + tw // 2, ty + th // 2
            cv2.line(frame, (cx, cy), (obj_cx, obj_cy), (0, 255, 255), 1)

        # Crosshairs at frame center
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 1)

        # HUD text
        cv2.putText(
            frame,
            f"FPS:{fps} BAT:{battery}%",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.putText(
            frame,
            f"YAW:{yaw:+4d} UD:{ud:+4d} FB:{fb:+4d}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.putText(
            frame,
            f"AREA:{(target[2]*target[3]) if target else 0}  TGT:{TARGET_AREA}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 0),
            1,
        )
        if self._recording:
            cv2.putText(
                frame, "[REC]", (frame.shape[1] - 70, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )
        cv2.putText(
            frame,
            "Q=quit L=land E=EMERG R=rec S=shot",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 200, 200),
            1,
        )


# ── Entry point ──────────────────────────────────────────────────

# Module-level reference so signal handlers can reach the drone.
_active_drone: DroneManager | None = None


def _shutdown_handler(signum: int, _frame: object) -> None:
    """Handle SIGTERM/SIGINT — land the drone and exit."""
    sig_name = signal.Signals(signum).name
    logger.warning("Received %s — landing drone and exiting", sig_name)
    if _active_drone is not None:
        state = _active_drone.state
        if state in (DroneState.FLYING, DroneState.LANDING):
            try:
                _active_drone.land()
            except Exception:
                _active_drone.emergency_stop()
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Tello visual tracker")
    parser.add_argument(
        "--ip",
        default=os.getenv("TELLO_IP", "192.168.10.1"),
        help="Tello IP address (default: $TELLO_IP or 192.168.10.1)",
    )
    parser.add_argument(
        "--no-fly",
        action="store_true",
        help="Debug mode: connect and stream video without takeoff",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed tracker logging (errors, area, commands)",
    )
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    drone = DroneManager(tello_ip=args.ip)
    _active_drone = drone

    with drone:
        logger.info("Loading YOLO model...")
        tracker = DroneTracker(drone)
        logger.info("Model loaded — ready")

        if not args.no_fly:
            drone.takeoff()
            logger.info("Drone airborne — starting tracker")

        try:
            tracker.run()
        except Exception as e:
            logger.critical("Unhandled exception in tracker: %s", e)
        finally:
            if drone.state == DroneState.FLYING:
                logger.info("Safety landing after tracker exit")
                try:
                    drone.land()
                except Exception:
                    drone.emergency_stop()
