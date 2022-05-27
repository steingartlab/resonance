import numpy as np
import pandas as pd

# Calibration values for FSR sensor
mass = np.array([1.4, 4.0, 5.9, 9.7])  # Weight put on sensor [kg]
voltage_raw = np.array([320, 545, 650, 777])  # Raw output
calibration_conductance = [4.56e-06, 1.14e-05, 1.74e-05, 3.16e-05]  # Calculated in calibration. 5V pin.
calibration_force = mass * 9.8

def get_pressure_function():
    """Generates a function from calibration data to map conductance to voltage.
    
    Returns:
        (class np.poly1D): A function to which experimental values can be passed.
    """

    coef = np.polyfit(
        x=calibration_conductance,
        y=calibration_force,
        deg=1
    )

    return np.poly1d(coef)

def get_resistance(V_out: np.array, V_in: int, R_r: float =  1e5):
    """Extracts resistance from FSR sensor output voltage.
    
    Args:
        V_out (np.array): Parsed voltage, see parse_voltage() output
        V_in (int): The microcontroller input voltage (either 3 or 5)
        R_r (float): Optional. Resistor resistance [Ohms]. Defaults to 1e5

    Returns:
        resistance (np.array)
    """
    return R_r * (V_in / V_out - 1)

def parse_voltage(V_out: pd.DataFrame, V_in) -> pd.DataFrame:
    """Scales voltage from 0-1023 to 0-V_in.

    Args:
        V_out (list-like): Raw V_out (i.e. as-is from Drops, incl. timestamps)
        V_in (int or array like): Optional. The microcontroller input voltage (either 3 or 5).
            Defaults to 5.

    Returns:
        (pd.DataFrame)
    """

    return V_in * (V_out + 1) / 1024


def pressure_from_raw_voltage(voltage_raw: pd.DataFrame, tducer_radius: float = 0.01, col: str = 'voltage', V_in=5) -> pd.DataFrame:
    """Transforms raw voltage from Arduino-connected FSR pressure sensor to pressure.
    
    Arduinos are 10-bit so the raw output (if not handled on the microcontroller)
    is going to be scaled between 0 and 1023. This parses the raw value by rescaling
    from 0 to the input voltage, maps that to resistance which can be converted to pressure
    after having calibrated the sensor.

    Args:
        voltage_raw (pd.DataFrame): Raw sensor data (i.e. as-is from Drops, incl. timestamps)
        t_ducer_radius (float): Optional. Transducer radius [m]. Defaults to 0.01
        col (str): Optional. Column name where voltage_raw resides. This only needs to be
            touched if doing constant pressure. Defaults to voltage.
        V_in (array-like): Input voltage of each sensor

    Returns:
        pressure (pd.DataFrame): Timestamped transducer pressure.
    """

    voltage_parsed = parse_voltage(V_out=voltage_raw, V_in=V_in)
    resistance = get_resistance(V_out=voltage_parsed, V_in=V_in)
    conductance = 1 / resistance
    pressure_function = get_pressure_function()
    pressure = conductance.copy()
    tducer_area = tducer_radius**2 * np.pi
    for column in pressure:
        pressure[column] = pressure_function(conductance)
        pressure[column] = pressure[column] / tducer_area

    return pressure 