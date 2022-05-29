import pandas as pd
import pytz
import sqlite3

import resonance.memory as memory
import dsp
import utils


EST = pytz.timezone('US/Eastern')

def _adjust(df: pd.DataFrame, base_col: str) -> pd.DataFrame:
    """Adjusts for sensor calibration differences by setting intial values as equal.
    
    Args:
        df (pd.DataFrame): Any multisensor data
        base_col (str): The reference column in df

    Returns:
        (pd.DataFrame)
    """
    
    base_val = df[base_col].iloc[0]
    for column in df:
        shift = base_val - df[column].iloc[0]
        df[column] = df[column] + shift

    return df

def _type_parse(df: pd.DataFrame) -> pd.DataFrame:
    """Parses df data (objects) to float.
    
    Helper function for query_arduino_data(). It's hacky, but bear with me.
    pd.read_sql() doesn't handle type-parsing well so this extra-step is needed.

    Args:
        df (pd.DataFrame): Any dataframe with all columns to be parsed as floats
    """
    for column in df:
        try:
            df[column] = df[column].astype(float)
        except ValueError:
            pass

    return df

def get_step_change_indices(df: pd.DataFrame, column: str, threshold: int = 3) -> list:
    """Gets indices where values change abruptly.

    Useful e.g. to highlight current switching on and off.
    
    Args:
        df (pd.DataFrame): Dataframe to be inspected
        column (str): The column name
        threshold (int): Optional. The minimum, absolute value of the difference
            between value t and t+1. Defaults to 3

    Returns:
        (list): A boolean list of the indices where the df switches phase
    """

    original = df[column].values
    shifted = df[column].shift(periods=1).values
    diff = original - shifted

    return [i for i, element in enumerate(diff) if abs(element)>threshold]

def parse_temperature_data(data: pd.DataFrame) -> pd.DataFrame:
    """Parses temperature sensing data to use for analysis.
    
    Double-thermistor data can be a little messy.
    
    Args:
        data (pd.DataFrame): Raw Arduino temperature sensing data

    Returns:
        (pd.DataFrame): Parse Arduino temperature sensing data
    """

    temperature_data = data.filter(items=['T_battery_1', 'T_battery_2'])
    temperature_data = dsp.remove_outliers(temperature_data)
    
    return _adjust(temperature_data, base_col='T_battery_1')

def query_arduino_data(name, exp_name: str, additional_query_params: str = None) -> pd.DataFrame:
    """Performs a sql query to pull and parse Arduino data from Drops.
    
    Args:
        name (str): The type of data.
        exp_name (str): The experiment name.
        additional_query_params (str): Optional. For more complicated queries.
            Defaults to None.

    Returns:
        (pd.DataFrame): A dataframe of the Arduino data

    Example:
        temperature_data = query_arduino_data(name='temperature', exp_name='Kokam-001-2022-03-20')
    """

    print(f'querying {name} . . .\n')

    db_location = f'/drops/data/brix5/{name}/{name}.sqlite3'
    connection = sqlite3.connect(db_location)
    query = f"SELECT * FROM '{exp_name}' {additional_query_params}"

    arduino_data_raw = pd.read_sql(
        sql=query,
        con=connection,
        parse_dates='time',
        index_col='time'
    )

    # Make timezone-aware to sync w other data
    arduino_data_raw.index = arduino_data_raw.index.tz_localize(pytz.utc).tz_convert(EST)

    arduino_data_parsed = _type_parse(df=arduino_data_raw)

    return arduino_data_parsed