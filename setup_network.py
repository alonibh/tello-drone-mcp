"""
setup_network.py — One-time script to switch the DJI Tello from AP mode to Station Mode.

Prerequisites:
  1. Power on the Tello (wait for the yellow blinking LED).
  2. Connect your computer to the Tello's Wi-Fi network (e.g. "TELLO-XXXXXX").
  3. Run this script: uv run python setup_network.py

After running, the Tello will reboot and join your local access point.
"""

from djitellopy import Tello

SSID = "Tello-Router"
PASSWORD = "BenHorin123"


def setup_station_mode() -> None:
    print("=" * 60)
    print("  Tello Station Mode Setup")
    print("=" * 60)
    print()
    print(f"  Target AP:   {SSID}")
    print(f"  Password:    {PASSWORD}")
    print()
    print("Connecting to Tello on its default AP (192.168.10.1)...")

    tello = Tello()
    try:
        tello.connect()
        battery = tello.get_battery()
        print(f"Connected! Battery: {battery}%")
        print()
        print(f'Sending station-mode command: ap "{SSID}" "{PASSWORD}"')
        result = tello.send_command_with_return(f"ap {SSID} {PASSWORD}")
        print(f"Response: {result}")
    except Exception as e:
        # A timeout after the AP command is expected — the Tello reboots
        # and drops its own AP before it can send an ACK.
        print(f"Command sent (drone likely rebooted): {e}")
    finally:
        try:
            tello.end()
        except Exception:
            pass

    _print_next_steps()


def _print_next_steps() -> None:
    print()
    print("=" * 60)
    print("  NEXT STEPS")
    print("=" * 60)
    print()
    print("  1. Wait ~10 seconds for the Tello to reboot.")
    print()
    print(f'  2. Connect your computer to the "{SSID}" Wi-Fi network.')
    print()
    print("  3. Find the Tello's new IP address on your router.")
    print("     Check your router's DHCP lease table for a new client")
    print("     (Tello MAC addresses typically start with 60:60:1F).")
    print()
    print("  4. Set the TELLO_IP environment variable to the new IP:")
    print("       export TELLO_IP=<new_ip>")
    print()
    print("  5. Start the MCP server:")
    print("       uv run python mcp_server.py")
    print()
    print("  To revert to AP mode, power-cycle the Tello while holding")
    print("  the power button for 5+ seconds (factory reset).")
    print("=" * 60)


if __name__ == "__main__":
    setup_station_mode()
