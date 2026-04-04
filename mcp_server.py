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
async def execute_flight_commands(commands: list[dict]) -> str:
    """Execute one or more drone flight commands. This is the ONLY tool for
    controlling drone movement. Use it for everything: single actions like
    taking off, AND multi-step sequences like "take off, go up, hover, land".

    Pass a JSON list of command objects. Each object needs an "action" key.
    Some actions also need a "value" key.

    Supported actions:
      {"action": "takeoff"}               — lift off and hover (no value needed)
      {"action": "land"}                  — land the drone (no value needed)
      {"action": "move_up", "value": N}   — ascend N centimeters (20–500). Convert meters to cm first!
      {"action": "move_down", "value": N} — descend N centimeters (20–500). Convert meters to cm first!
      {"action": "rotate", "value": N}    — rotate N degrees (-360 to 360). Positive = clockwise.
      {"action": "hover", "value": N}     — hold position for N seconds (1–120)

    Single command example: [{"action": "takeoff"}]
    Multi-step example:     [{"action": "takeoff"}, {"action": "move_up", "value": 400},
                             {"action": "hover", "value": 5}, {"action": "land"}]

    If any step fails, execution stops and the error is returned along with
    which steps completed successfully.

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
                if drone.state != DroneState.CONNECTED:
                    return f"{step} failed: drone not ready (state={drone.state.name}). Completed: {'; '.join(results)}"
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
