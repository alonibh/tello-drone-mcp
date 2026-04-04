"""
tello_controller.py — Core DroneManager class for the DJI Tello.

Handles UDP connection, flight commands, background video capture,
and battery monitoring with automatic safety landing.
"""

import logging
import threading
import time
from enum import Enum, auto
import numpy as np
from djitellopy import Tello

logger = logging.getLogger("tello.controller")

BATTERY_WARN_PCT = 20
BATTERY_LAND_PCT = 10
BATTERY_POLL_SEC = 15
RECONNECT_RETRIES = 3
RECONNECT_DELAY = 3.0
MOVE_MIN_CM = 20
MOVE_MAX_CM = 200


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
        self._state_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._video_thread: threading.Thread | None = None
        self._battery_thread: threading.Thread | None = None
        self._last_battery_pct: int = -1

    # ── Context manager ──────────────────────────────────────────

    def __enter__(self) -> "DroneManager":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()

    # ── Lifecycle ────────────────────────────────────────────────

    def connect(self) -> None:
        self._stop_background_threads()
        for attempt in range(1, RECONNECT_RETRIES + 1):
            try:
                self._tello.connect()
                with self._state_lock:
                    self._state = DroneState.CONNECTED
                self._start_background_threads()
                self._last_battery_pct = self._tello.get_battery()
                logger.info(
                    "Connected to Tello at %s (battery %d%%)",
                    self._ip,
                    self._last_battery_pct,
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

        with self._state_lock:
            self._state = DroneState.ERROR
        raise ConnectionError(
            f"Could not connect to Tello at {self._ip} "
            f"after {RECONNECT_RETRIES} attempts"
        )

    def disconnect(self) -> None:
        self._stop_event.set()
        need_land = False
        with self._state_lock:
            if self._state in (DroneState.FLYING, DroneState.LANDING):
                need_land = True
                self._state = DroneState.LANDING
        if need_land:
            logger.info("Auto-landing before disconnect")
            try:
                self._tello.land()
            except Exception:
                try:
                    self._tello.emergency()
                except Exception:
                    pass
        self._join_background_threads()
        try:
            self._tello.streamoff()
        except Exception:
            pass
        try:
            self._tello.end()
        except Exception:
            pass
        with self._state_lock:
            self._state = DroneState.DISCONNECTED
        logger.info("Disconnected from Tello")

    # ── Flight commands ──────────────────────────────────────────

    def takeoff(self) -> None:
        with self._state_lock:
            self._require_state(DroneState.CONNECTED)
        self._tello.takeoff()
        with self._state_lock:
            self._state = DroneState.FLYING
        logger.info("Takeoff complete")

    def land(self) -> None:
        with self._state_lock:
            if self._state == DroneState.LANDING:
                return
            self._require_state(DroneState.FLYING)
            self._state = DroneState.LANDING
        try:
            self._tello.land()
        except Exception:
            with self._state_lock:
                self._state = DroneState.ERROR
            raise
        with self._state_lock:
            self._state = DroneState.CONNECTED
        logger.info("Landing complete")

    def emergency_stop(self) -> None:
        """Cut all motors immediately."""
        logger.critical("EMERGENCY STOP — killing motors")
        try:
            self._tello.emergency()
        except Exception:
            pass
        with self._state_lock:
            self._state = DroneState.ERROR

    def send_rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        """Send RC control only if the drone is currently FLYING."""
        with self._state_lock:
            if self._state != DroneState.FLYING:
                return
        self._tello.send_rc_control(lr, fb, ud, yaw)

    def move_up(self, cm: int) -> None:
        with self._state_lock:
            self._require_state(DroneState.FLYING)
        cm = max(MOVE_MIN_CM, min(MOVE_MAX_CM, cm))
        self._tello.move_up(cm)

    def move_down(self, cm: int) -> None:
        with self._state_lock:
            self._require_state(DroneState.FLYING)
        cm = max(MOVE_MIN_CM, min(MOVE_MAX_CM, cm))
        self._tello.move_down(cm)

    def rotate_clockwise(self, degrees: int) -> None:
        with self._state_lock:
            self._require_state(DroneState.FLYING)
        self._tello.rotate_clockwise(degrees)

    def rotate_counter_clockwise(self, degrees: int) -> None:
        with self._state_lock:
            self._require_state(DroneState.FLYING)
        self._tello.rotate_counter_clockwise(degrees)

    # ── Frame access ─────────────────────────────────────────────

    def get_latest_frame(self) -> np.ndarray | None:
        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    # ── Background threads ───────────────────────────────────────

    def _stop_background_threads(self) -> None:
        """Signal background threads to stop and wait for them."""
        self._stop_event.set()
        self._join_background_threads()

    def _join_background_threads(self) -> None:
        """Wait for background threads to finish (with timeout)."""
        if self._video_thread is not None and self._video_thread.is_alive():
            self._video_thread.join(timeout=2.0)
        if self._battery_thread is not None and self._battery_thread.is_alive():
            self._battery_thread.join(timeout=2.0)

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
                        self._latest_frame = frame.copy()
            except Exception as e:
                logger.debug("Video frame error: %s", e)
            time.sleep(0.01)

    def _battery_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                pct = self._tello.get_battery()
                self._last_battery_pct = pct
                with self._state_lock:
                    is_flying = self._state == DroneState.FLYING
                if pct <= BATTERY_LAND_PCT and is_flying:
                    logger.critical("Battery %d%% — emergency auto-land", pct)
                    try:
                        self.land()
                    except RuntimeError:
                        pass
                    except Exception:
                        logger.warning(
                            "Graceful land failed in battery loop, trying emergency stop"
                        )
                        self.emergency_stop()
                elif pct <= BATTERY_WARN_PCT:
                    logger.warning("Battery low: %d%%", pct)
            except Exception as e:
                logger.warning("Battery poll error: %s", e)
            self._stop_event.wait(BATTERY_POLL_SEC)
        logger.info("Battery monitor thread exiting")

    # ── Helpers ──────────────────────────────────────────────────

    def _require_state(self, required: DroneState) -> None:
        if self._state != required:
            raise RuntimeError(
                f"Command requires state {required.name}, "
                f"current state is {self._state.name}"
            )

    @property
    def state(self) -> DroneState:
        with self._state_lock:
            return self._state

    @property
    def ip(self) -> str:
        return self._ip

    @property
    def battery(self) -> int:
        return self._last_battery_pct

    @property
    def tello(self) -> Tello:
        return self._tello
