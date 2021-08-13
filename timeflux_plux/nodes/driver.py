import platform
import sys
import os
import numpy as np
import pandas as pd
from time import time, sleep
from threading import Thread, Lock
from timeflux.core.node import Node
from timeflux.core.exceptions import WorkerInterrupt
import timeflux_plux.helpers.transfer as transfer

# Load library according to system
_lib = platform.system()
_lib += "ARM" if platform.machine().startswith("arm") else ""
_lib += "64" if sys.maxsize > 2 ** 32 else "32"
if platform.system() == "Windows":
    _lib += "_37" if sys.version.startswith("3.7") else "_38"
if platform.system() == "Darwin":
    if sys.version.startswith("3.7"):
        _lib += "_37"
    elif sys.version.startswith("3.8"):
        _lib += "_38"
    else:
        _lib += "_39"
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "libs", _lib))
import plux

# Sensor types
SENSORS = [
    "UNKNOWN",
    "EMG",
    "ECG",
    "LUX",
    "EDA",
    "BVP",
    "RESP",
    "XYZ",
    "SYNC",
    "EEG",
    "SYNC_ADAP",
    "SYNC_LED",
    "SYNC_SW",
    "USB",
    "FORCE",
    "TEMP",
    "VPROBE",
    "BREAKOUT",
    "SpO2",
    "GONI",
    "ACT",
    "EOG",
    "EGG",
]

# Maximum rates
RATES = [0, 8000, 5000, 4000, 3000, 3000, 2000, 2000, 2000]

# Available transfer functions
TRANSFER = [f for f in dir(transfer) if not f.startswith("_")]


class Plux(Node):

    """
    This node connects to a BiosignalsPlux device and streams data at a provided rate.

    Two output streams are provided. The default output is the data read from the analog
    and digital channels, converted to meaningful units according to the sensor types.
    The ``o_raw`` output provides the data directly returned from the device.

    Args:
        port (string|None): Path to the Plux device.
            e.g. `xx:xx:xx:xx:xx:xx` (Bluetooth Mac Address), `COMx` (serial port on Windows), `/dev/cu.biosignalsplux-Bluetoot` (serial port on macOS).
            If not specified, the node will connect to the first detected device.
        rate (int|None): The device rate in Hz.
            Maximum value for one channel: `8000`.
            Maximum value for eight channels: `2000`.
            If not specified, the rate will be set to the maximum value allowed for the number of detected sensors.

    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Signal converted to meaningful units, provides DataFrame.
        o_raw (Port): Raw signal, provides DataFrame.

    Example:
        .. literalinclude:: /../examples/plux.yaml
            :language: yaml

    .. attention::

        * On macOS, device autodetection and MAC addresses seem to work, but data is not actually streamed. Use the serial port instead.
        * Multiple sensors of the same type are currenly not supported.
        * For sensors that return multiple channels (accelerator for example), only the first channel is available.

    .. seealso::

        * `Official (outdated) API documentation <https://biosignalsplux.com/downloads/apis/python-api-docs/>`_
        * `Official libraries <https://github.com/biosignalsplux/python-samples/tree/master/PLUX-API-Python3>`_
        * `Discussion about sensor detection mapping <https://github.com/biosignalsplux/python-samples/issues/8>`_
        * `Helpful examples on how to write transfer functions <https://github.com/biosignalsplux/biosignalsnotebooks>`_
    """

    def __init__(self, address=None, rate=None):

        # Find the first available device
        if address is None:
            devices = plux.BaseDev.findDevices()
            for device in devices:
                if device[1] == "biosignalsplux":
                    address = device[0]
                    break

        # No device found
        if address is None:
            raise WorkerInterrupt("No device found")

        # Validate address
        if not isinstance(address, str):
            self.logger.warn("Invalid address")

        # Initialize device and display information
        self.device = Device(address)
        self.logger.info(f"Connected to {address}")
        self.logger.info(self.info())

        # Get channels
        sensors = self.device.getSensors()
        channels = []
        mask = ["0"] * max(sensors.keys())
        for channel, sensor in sensors.items():
            mask[-channel] = "1"
            channels.append(SENSORS[sensor.clas])
        mask = int("".join(mask), 2)

        # Validate rate
        if rate is not None:
            if not isinstance(rate, int) or rate > RATES[len(channels)]:
                self.logger.warn("Invalid rate")
                rate = None

        # Get maximum rate
        if rate is None:
            rate = RATES[len(channels)]
            self.logger.info(f"Setting rate to {rate}")

        # Set attributes
        self.channels = channels
        self.delta = pd.Timedelta(1 / rate, "second")
        self.meta = {"rate": rate}
        self.functions = {}

        # Map channels to transfer functions
        for index, channel in enumerate(channels):
            for function in TRANSFER:
                if channel.upper().startswith(function):
                    self.functions[index] = function

        # Start data acquisition in a new thread
        self.device.start(rate, mask, 16)
        self.device.lock = Lock()
        self.thread = Thread(target=self.device.loop).start()

    def update(self):
        """Update outputs"""

        # Lock the thread and copy data
        self.device.lock.acquire()
        indices = np.array(self.device.indices, dtype=int)
        samples = np.array(self.device.samples, dtype=float)
        missed = self.device.missed
        self.device.missed = 0
        self.device.indices = []
        self.device.samples = []
        self.device.lock.release()

        # Prepare the data for output
        if missed > 0:
            self.logger.warning(f"Missed {missed} sample(s)")
        if len(samples) > 0:
            indices *= self.delta
            indices += self.device.time
            self.o.set(self.convert(samples), indices, self.channels, self.meta)
            self.o_raw.set(samples, indices, self.channels, self.meta)

    def terminate(self):
        """Cleanup"""
        self.device.exit = True
        sleep(0.1)
        self.device.stop()
        self.device.close()

    def info(self):
        """Get some info about the connected device"""
        properties = self.device.getProperties()
        fw_version = properties["fwVersion"].to_bytes(2, "big")
        hw_version = properties["hwVersion"].to_bytes(2, "big")
        return {
            "name": properties["uid"],
            "firmware_version": f"{fw_version[0]}.{fw_version[1]}",
            "hardware_version": f"{hw_version[0]}.{hw_version[1]}",
            "battery_level": f"{self.device.getBattery()}%",
        }

    def convert(self, samples):
        """Convert signal to meaningful units"""

        # No transfer function found
        if not self.functions:
            return samples

        # Apply transfer function to each identified channel
        signals = samples.copy()
        for channel, function in self.functions.items():
            signals[:, channel] = getattr(transfer, function)(signals[:, channel])
        return signals


class Device(plux.SignalsDev):

    """Plux device"""

    def __init__(self, address):
        plux.MemoryDev.__init__(address)
        self.time = None
        self.exit = False
        self.counter = 0
        self.missed = 0
        self.indices = []
        self.samples = []

    def onRawFrame(self, counter, data):
        self.lock.acquire()
        if not self.time:
            self.time = pd.Timestamp(time(), unit="s")
        self.missed += counter - self.counter - 1
        self.counter = counter
        self.indices.append(counter)
        self.samples.append(data)
        self.lock.release()
        return self.exit


__all__ = ["Plux"]
