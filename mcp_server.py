"""
mcp_server.py — MCP server exposing DJI Tello drone controls as LLM tools.

Usage:
  uv run python mcp_server.py
"""

import asyncio
import base64
import logging
import os

import cv2
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image

from tello_controller import DroneManager, DroneState

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("tello.mcp")

mcp = FastMCP("tello-drone")
drone = DroneManager(tello_ip=os.getenv("TELLO_IP", "192.168.10.1"))

_DISCONNECTED_MSG = (
    "Error: drone is not connected. Call the connect_drone tool first."
)
_NOT_FLYING_MSG = "Error: drone must be flying to execute this command (state={state})."


@mcp.tool()
async def connect_drone() -> str:
    """Establish the UDP connection to the Tello drone and start video/battery threads.

    This must be called before any other drone tool. The drone should be powered
    on and your computer connected to its Wi-Fi network (default IP 192.168.10.1).
    """
    if drone.state in (DroneState.CONNECTED, DroneState.FLYING):
        return f"Drone is already connected (state={drone.state.name})."
    try:
        drone.connect()
        battery = drone.tello.get_battery()
        return (
            f"Connected to Tello at {drone.tello.address[0]}. "
            f"Battery: {battery}%. Video stream active."
        )
    except ConnectionError as e:
        return f"Connection failed: {e}"


@mcp.tool()
async def takeoff() -> str:
    """Command the Tello drone to take off and hover at ~1 meter.

    Takes absolutely no arguments. Do not pass speed, degrees, or battery.
    """
    if drone.state == DroneState.DISCONNECTED:
        return _DISCONNECTED_MSG
    if drone.state == DroneState.FLYING:
        return "Error: drone is already flying."
    if drone.state != DroneState.CONNECTED:
        return f"Error: drone is not ready (state={drone.state.name})."
    try:
        drone.takeoff()
        return "Takeoff successful. Drone is now hovering."
    except Exception as e:
        return f"Takeoff failed: {e}"


@mcp.tool()
async def land() -> str:
    """Command the Tello drone to land safely."""
    if drone.state == DroneState.DISCONNECTED:
        return _DISCONNECTED_MSG
    if drone.state != DroneState.FLYING:
        return f"Error: drone is not flying (state={drone.state.name})."
    try:
        drone.land()
        return "Landing successful. Drone is grounded."
    except Exception as e:
        return f"Landing failed: {e}"


@mcp.tool()
async def rotate(degrees: int) -> str:
    """Rotate the drone by the given number of degrees.

    Positive values rotate clockwise, negative values rotate counter-clockwise.

    Args:
        degrees: Rotation amount between -360 and 360.
    """
    if drone.state == DroneState.DISCONNECTED:
        return _DISCONNECTED_MSG
    if not isinstance(degrees, int):
        return "Error: degrees must be an integer."
    if not (-360 <= degrees <= 360):
        return "Error: degrees must be between -360 and 360."
    if degrees == 0:
        return "No rotation needed (0 degrees)."
    if drone.state != DroneState.FLYING:
        return _NOT_FLYING_MSG.format(state=drone.state.name)
    try:
        if degrees > 0:
            drone.rotate_clockwise(degrees)
        else:
            drone.rotate_counter_clockwise(abs(degrees))
        direction = "clockwise" if degrees > 0 else "counter-clockwise"
        return f"Rotated {direction} by {abs(degrees)} degrees."
    except Exception as e:
        return f"Rotate failed: {e}"


@mcp.tool()
async def move_up(cm: int) -> str:
    """Move the drone upward by the specified number of centimeters.

    IMPORTANT: The value must be in centimeters, not meters.
    If the user says "go up 4 meters", you must pass cm=400.
    Valid range: 20–500 cm.

    Args:
        cm: Distance to move up in centimeters (20–500).
    """
    if drone.state == DroneState.DISCONNECTED:
        return _DISCONNECTED_MSG
    if drone.state != DroneState.FLYING:
        return _NOT_FLYING_MSG.format(state=drone.state.name)
    try:
        drone.move_up(cm)
        return f"Moved up {cm} cm."
    except Exception as e:
        return f"Move up failed: {e}"


@mcp.tool()
async def move_down(cm: int) -> str:
    """Move the drone downward by the specified number of centimeters.

    IMPORTANT: The value must be in centimeters, not meters.
    If the user says "go down 2 meters", you must pass cm=200.
    Valid range: 20–500 cm.

    Args:
        cm: Distance to move down in centimeters (20–500).
    """
    if drone.state == DroneState.DISCONNECTED:
        return _DISCONNECTED_MSG
    if drone.state != DroneState.FLYING:
        return _NOT_FLYING_MSG.format(state=drone.state.name)
    try:
        drone.move_down(cm)
        return f"Moved down {cm} cm."
    except Exception as e:
        return f"Move down failed: {e}"


@mcp.tool()
async def hover(seconds: int) -> str:
    """Hold the drone's current position for the specified number of seconds.

    The drone must already be flying. This simply waits without sending
    any movement commands, so the drone hovers in place.

    Args:
        seconds: Duration to hover in seconds (1–120).
    """
    if drone.state == DroneState.DISCONNECTED:
        return _DISCONNECTED_MSG
    if drone.state != DroneState.FLYING:
        return _NOT_FLYING_MSG.format(state=drone.state.name)
    if not (1 <= seconds <= 120):
        return "Error: seconds must be between 1 and 120."
    await asyncio.sleep(seconds)
    return f"Hovered in place for {seconds} seconds."


@mcp.tool()
async def execute_flight_plan(commands: list[dict]) -> str:
    """Execute a sequence of drone commands in a single call.

    This is faster than calling tools one at a time. Pass a list of command
    objects. Each object must have an "action" key. Some actions require a
    "value" key.

    Supported actions and their values:
      - {"action": "takeoff"}              — take off (no value needed)
      - {"action": "land"}                 — land (no value needed)
      - {"action": "move_up", "value": N}  — move up N centimeters (20–500)
      - {"action": "move_down", "value": N} — move down N centimeters (20–500)
      - {"action": "rotate", "value": N}   — rotate N degrees (-360 to 360, positive=clockwise)
      - {"action": "hover", "value": N}    — hover in place for N seconds (1–120)

    Example: [{"action": "takeoff"}, {"action": "move_up", "value": 400},
              {"action": "hover", "value": 5}, {"action": "land"}]

    Args:
        commands: List of command dicts with "action" and optional "value" keys.
    """
    if drone.state == DroneState.DISCONNECTED:
        return _DISCONNECTED_MSG

    results: list[str] = []
    for i, cmd in enumerate(commands, 1):
        action = cmd.get("action")
        value = cmd.get("value")
        step = f"Step {i} ({action})"

        try:
            if action == "takeoff":
                if drone.state == DroneState.FLYING:
                    results.append(f"{step}: skipped, already flying.")
                    continue
                drone.takeoff()
                results.append(f"{step}: takeoff OK.")

            elif action == "land":
                if drone.state != DroneState.FLYING:
                    results.append(f"{step}: skipped, not flying.")
                    continue
                drone.land()
                results.append(f"{step}: land OK.")

            elif action == "move_up":
                if drone.state != DroneState.FLYING:
                    return f"{step} failed: drone is not flying. Completed: {'; '.join(results)}"
                drone.move_up(value)
                results.append(f"{step}: moved up {value} cm.")

            elif action == "move_down":
                if drone.state != DroneState.FLYING:
                    return f"{step} failed: drone is not flying. Completed: {'; '.join(results)}"
                drone.move_down(value)
                results.append(f"{step}: moved down {value} cm.")

            elif action == "rotate":
                if drone.state != DroneState.FLYING:
                    return f"{step} failed: drone is not flying. Completed: {'; '.join(results)}"
                if value > 0:
                    drone.rotate_clockwise(value)
                else:
                    drone.rotate_counter_clockwise(abs(value))
                results.append(f"{step}: rotated {value} degrees.")

            elif action == "hover":
                if drone.state != DroneState.FLYING:
                    return f"{step} failed: drone is not flying. Completed: {'; '.join(results)}"
                await asyncio.sleep(value)
                results.append(f"{step}: hovered {value} seconds.")

            else:
                return f"{step} failed: unknown action '{action}'. Completed: {'; '.join(results)}"

        except Exception as e:
            return f"{step} failed: {e}. Completed: {'; '.join(results)}"

    return "Flight plan complete. " + "; ".join(results)


@mcp.tool()
async def get_latest_camera_frame() -> Image:
    """Capture the latest video frame from the drone's camera.

    Returns a PNG image that can be analyzed for navigation,
    object detection, or situational awareness.
    """
    if drone.state == DroneState.DISCONNECTED:
        return _DISCONNECTED_MSG

    frame = drone.get_latest_frame()
    if frame is None:
        return "Error: no frame available yet. The camera may still be initializing."

    success, buffer = cv2.imencode(".png", frame)
    if not success:
        return "Error: failed to encode frame as PNG."

    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return Image(data=b64, format="png")


if __name__ == "__main__":
    logger.info("Starting Tello MCP server (stdio transport)")
    mcp.run(transport="stdio")
