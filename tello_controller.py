"""
tello_controller.py — Core DroneManager class for the DJI Tello.

Handles UDP connection, flight commands, background video capture,
and battery monitoring with automatic safety landing.
"""

import logging
import threading
import time
from enum import Enum, auto
from typing import Optional

import numpy as np
from djitellopy import Tello

logger = logging.getLogger("tello.controller")

BATTERY_WARN_PCT = 20
BATTERY_LAND_PCT = 10
BATTERY_POLL_SEC = 15
RECONNECT_RETRIES = 3
RECONNECT_DELAY = 3.0
MOVE_MIN_CM = 20
MOVE_MAX_CM = 500


class DroneState(Enum):
    DISCONNECTED = auto()
    CONNECTED = auto()
    FLYING = auto()
    LANDING = auto()
    ERROR = auto()


class DroneManager:
    """Thread-safe controller for a single DJI Tello drone."""

    def __init__(self, tello_ip: str = "192.168.10.1") -> None:
        self._ip = tello_ip
        self._tello = Tello(host=tello_ip)
        self._state = DroneState.DISCONNECTED
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._video_thread: Optional[threading.Thread] = None
        self._battery_thread: Optional[threading.Thread] = None

    # ── Context manager ──────────────────────────────────────────

    def __enter__(self) -> "DroneManager":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()

    # ── Lifecycle ────────────────────────────────────────────────

    def connect(self) -> None:
        for attempt in range(1, RECONNECT_RETRIES + 1):
            try:
                self._tello.connect()
                self._state = DroneState.CONNECTED
                self._start_background_threads()
                battery = self._tello.get_battery()
                logger.info(
                    "Connected to Tello at %s (battery %d%%)", self._ip, battery
                )
                return
            except Exception as e:
                logger.warning(
                    "Connect attempt %d/%d failed: %s",
                    attempt,
                    RECONNECT_RETRIES,
                    e,
                )
                if attempt < RECONNECT_RETRIES:
                    time.sleep(RECONNECT_DELAY)

        self._state = DroneState.ERROR
        raise ConnectionError(
            f"Could not connect to Tello at {self._ip} "
            f"after {RECONNECT_RETRIES} attempts"
        )

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._state == DroneState.FLYING:
            logger.info("Auto-landing before disconnect")
            try:
                self._tello.land()
            except Exception:
                pass
        try:
            self._tello.streamoff()
        except Exception:
            pass
        try:
            self._tello.end()
        except Exception:
            pass
        self._state = DroneState.DISCONNECTED
        logger.info("Disconnected from Tello")

    # ── Flight commands ──────────────────────────────────────────

    def takeoff(self) -> None:
        self._require_state(DroneState.CONNECTED)
        self._tello.takeoff()
        self._state = DroneState.FLYING
        logger.info("Takeoff complete")

    def land(self) -> None:
        self._require_state(DroneState.FLYING)
        self._state = DroneState.LANDING
        self._tello.land()
        self._state = DroneState.CONNECTED
        logger.info("Landing complete")

    def move_up(self, cm: int) -> None:
        self._require_state(DroneState.FLYING)
        cm = max(MOVE_MIN_CM, min(MOVE_MAX_CM, cm))
        self._tello.move_up(cm)

    def move_down(self, cm: int) -> None:
        self._require_state(DroneState.FLYING)
        cm = max(MOVE_MIN_CM, min(MOVE_MAX_CM, cm))
        self._tello.move_down(cm)

    def rotate_clockwise(self, degrees: int) -> None:
        self._require_state(DroneState.FLYING)
        self._tello.rotate_clockwise(degrees)

    def rotate_counter_clockwise(self, degrees: int) -> None:
        self._require_state(DroneState.FLYING)
        self._tello.rotate_counter_clockwise(degrees)

    # ── Frame access ─────────────────────────────────────────────

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    # ── Background threads ───────────────────────────────────────

    def _start_background_threads(self) -> None:
        self._tello.streamon()
        self._stop_event.clear()

        self._video_thread = threading.Thread(
            target=self._video_loop, daemon=True, name="tello-video"
        )
        self._video_thread.start()

        self._battery_thread = threading.Thread(
            target=self._battery_loop, daemon=True, name="tello-battery"
        )
        self._battery_thread.start()

    def _video_loop(self) -> None:
        frame_read = self._tello.get_frame_read()
        while not self._stop_event.is_set():
            try:
                frame = frame_read.frame
                if frame is not None:
                    with self._frame_lock:
                        self._latest_frame = frame
            except Exception as e:
                logger.debug("Video frame error: %s", e)
            time.sleep(0.01)

    def _battery_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                pct = self._tello.get_battery()
                if pct <= BATTERY_LAND_PCT and self._state == DroneState.FLYING:
                    logger.critical("Battery %d%% — emergency auto-land", pct)
                    try:
                        self.land()
                    except Exception:
                        pass
                elif pct <= BATTERY_WARN_PCT:
                    logger.warning("Battery low: %d%%", pct)
            except Exception as e:
                logger.debug("Battery poll error: %s", e)
            self._stop_event.wait(BATTERY_POLL_SEC)

    # ── Helpers ──────────────────────────────────────────────────

    def _require_state(self, required: DroneState) -> None:
        if self._state != required:
            raise RuntimeError(
                f"Command requires state {required.name}, "
                f"current state is {self._state.name}"
            )

    @property
    def state(self) -> DroneState:
        return self._state

    @property
    def tello(self) -> Tello:
        return self._tello
