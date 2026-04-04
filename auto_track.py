"""
auto_track.py — Autonomous visual tracking for the DJI Tello.

Uses Haar Cascade (or optional YOLOv8) to detect a target in the drone's
video feed and a PID controller to keep the target centered via continuous
yaw and altitude adjustments.

Usage:
  uv run python auto_track.py --detector face
  uv run python auto_track.py --detector body --no-fly   # debug without takeoff
  uv run python auto_track.py --detector yolo             # requires: uv add ultralytics
"""

import argparse
import logging
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np

from tello_controller import DroneManager, DroneState

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("tello.tracker")

# ── PID Controller ───────────────────────────────────────────────

YAW_PID_GAINS = (0.25, 0.0, 0.10)
ALT_PID_GAINS = (0.20, 0.0, 0.08)


class PIDController:
    """Simple time-based PID controller with output clamping."""

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

FACE_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
BODY_CASCADE = cv2.data.haarcascades + "haarcascade_fullbody.xml"


class DroneTracker:
    """Autonomous visual tracker that keeps a detected target centered."""

    def __init__(self, drone: DroneManager, detector: str = "face") -> None:
        self._drone = drone
        self._detector = detector
        self._yaw_pid = PIDController(*YAW_PID_GAINS)
        self._alt_pid = PIDController(*ALT_PID_GAINS)
        self._cascade: Optional[cv2.CascadeClassifier] = None
        self._yolo = None

        if detector == "yolo":
            self._yolo = self._load_yolo()
        else:
            path = FACE_CASCADE if detector == "face" else BODY_CASCADE
            self._cascade = cv2.CascadeClassifier(path)
            if self._cascade.empty():
                raise RuntimeError(f"Failed to load cascade: {path}")

    @staticmethod
    def _load_yolo():
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "YOLOv8 requires ultralytics. Install with: uv add ultralytics"
            )
        return YOLO("yolov8n.pt")

    def _detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return list of (x, y, w, h) bounding boxes."""
        if self._detector == "yolo" and self._yolo is not None:
            results = self._yolo(frame, verbose=False)[0]
            boxes = []
            for r in results.boxes:
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                boxes.append((x1, y1, x2 - x1, y2 - y1))
            return boxes

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        if len(rects) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in rects]

    @staticmethod
    def _select_target(
        detections: list[tuple[int, int, int, int]],
    ) -> Optional[tuple[int, int, int, int]]:
        """Pick the largest bounding box by area."""
        if not detections:
            return None
        return max(detections, key=lambda b: b[2] * b[3])

    @staticmethod
    def _check_stdin() -> Optional[str]:
        """Return a character from stdin if available, without blocking."""
        if sys.platform == "win32":
            import msvcrt
            if msvcrt.kbhit():
                return msvcrt.getwch().lower()
        else:
            import select
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.readline().strip().lower()
        return None

    def run(self) -> None:
        """Main tracking loop. Press 'q' to quit, 'e' for emergency land."""
        print("Tracker started. Press 'q' to quit, 'e' for emergency land.")

        fps_timer = time.time()
        fps = 0
        frame_count = 0

        while True:
            key = cv2.waitKey(1) & 0xFF
            console = self._check_stdin()
            if key == ord("q") or console == "q":
                logger.info("Quit requested")
                break
            if key == ord("e") or console == "e":
                logger.warning("EMERGENCY STOP")
                if self._drone.state == DroneState.FLYING:
                    self._drone.land()
                break

            frame = self._drone.get_latest_frame()
            if frame is None:
                time.sleep(0.033)
                continue

            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2

            detections = self._detect(frame)
            target = self._select_target(detections)

            yaw_cmd = 0
            ud_cmd = 0

            if target is not None:
                tx, ty, tw, th = target
                obj_cx = tx + tw // 2
                obj_cy = ty + th // 2

                x_err = obj_cx - cx  # positive = object is right of center
                y_err = cy - obj_cy  # positive = object is above center -> go up

                yaw_cmd = int(self._yaw_pid.compute(x_err))
                ud_cmd = int(self._alt_pid.compute(y_err))
            else:
                self._yaw_pid.reset()
                self._alt_pid.reset()

            if self._drone.state == DroneState.FLYING:
                self._drone.tello.send_rc_control(0, 0, ud_cmd, yaw_cmd)

            # FPS counter
            frame_count += 1
            if time.time() - fps_timer >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_timer = time.time()

            self._draw_debug(frame, detections, target, cx, cy, yaw_cmd, ud_cmd, fps)
            cv2.imshow("Tello Tracker", frame)

        # Stop all RC movement and close window
        if self._drone.state == DroneState.FLYING:
            self._drone.tello.send_rc_control(0, 0, 0, 0)
        cv2.destroyAllWindows()

    def _draw_debug(
        self,
        frame: np.ndarray,
        detections: list[tuple[int, int, int, int]],
        target: Optional[tuple[int, int, int, int]],
        cx: int,
        cy: int,
        yaw: int,
        ud: int,
        fps: int,
    ) -> None:
        """Draw bounding boxes, crosshairs, and HUD overlay."""
        battery = 0
        try:
            battery = self._drone.tello.get_battery()
        except Exception:
            pass

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
            f"YAW:{yaw:+4d} UD:{ud:+4d}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.putText(
            frame,
            f"Detector: {self._detector}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 0),
            1,
        )
        cv2.putText(
            frame,
            "Q=quit  E=emergency land",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 200, 200),
            1,
        )


# ── Entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Tello visual tracker")
    parser.add_argument(
        "--ip",
        default=os.getenv("TELLO_IP", "192.168.10.1"),
        help="Tello IP address (default: $TELLO_IP or 192.168.10.1)",
    )
    parser.add_argument(
        "--detector",
        default="face",
        choices=["face", "body", "yolo"],
        help="Detection backend (default: face)",
    )
    parser.add_argument(
        "--no-fly",
        action="store_true",
        help="Debug mode: connect and stream video without takeoff",
    )
    args = parser.parse_args()

    with DroneManager(tello_ip=args.ip) as drone:
        if not args.no_fly:
            drone.takeoff()
            logger.info("Drone airborne — starting tracker")

        tracker = DroneTracker(drone, detector=args.detector)
        tracker.run()

        if drone.state == DroneState.FLYING:
            drone.land()
