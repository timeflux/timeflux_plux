"""Transfer functions"""

# Default resolution
RESOLUTION = 16

# Working voltage
VCC = 3


def ECG(signal):
    """ECG value in millivolt (𝑚𝑉)"""
    return ((signal / 2 ** RESOLUTION) - 0.5) * VCC


def BVP(signal):
    """BVP value in  r.i. units"""
    return signal / 2 ** RESOLUTION


def EDA(signal):
    """EDA value in microsiemens (𝜇𝑆)"""
    return ((signal / 2 ** RESOLUTION) * VCC) / 0.12


def EMG(signal):
    """EMG value in millivolt (𝑚𝑉)"""
    return ((signal / 2 ** RESOLUTION) - 0.5) * VCC


def PZT(signal):
    """Displacement value in percentage (%) of full scale"""
    return ((signal / 2 ** RESOLUTION) - 0.5) * 100


def EEG(signal):
    """EEG value in microvolt (𝜇𝑉)"""
    return ((((signal / 2 ** RESOLUTION) - 0.5) * VCC) / 40000) * 1e-6
