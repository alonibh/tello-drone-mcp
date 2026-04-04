"""
mcp_server.py — MCP server exposing DJI Tello drone controls as LLM tools.

Usage:
  uv run python mcp_server.py
"""

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
async def takeoff() -> str:
    """Command the Tello drone to take off and hover at ~1 meter."""
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
        return f"Error: drone must be flying to rotate (state={drone.state.name})."
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
async def get_latest_camera_frame() -> Image | str:
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
