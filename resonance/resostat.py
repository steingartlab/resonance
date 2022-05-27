import h5py
import numpy as np
import pandas as pd
import sqlite3

def _parse_waves(raw_data: pd.DataFrame) -> np.array:
    '''Parses waves from the raw data file.
    
    Pandas reads list-values as str. This resolves that issue by
    parsing them as np arrays.
    
    Returns:
        (np.array): waveforms w. shape of #sweepsX#data-points-per-sweep
    '''

    waves_formatted = raw_data['data'].str.strip('[]').str.split(',')
    waves = np.zeros((len(waves_formatted), len(waves_formatted[0])),
                        dtype=np.float16)  # np.float16 uses less memory
    for i, wave in enumerate(waves_formatted):
        waves[i, :] = wave

    return waves

def _get_frequencies(start_freq: int, end_freq: int, increment: int) -> np.array:
    '''Gets the frequencies used in experiment's sweep.

    Args:
        start_freq (int): The starting frequency [Hz]
        end_freq (int): The ending frequency [Hz]
        increment (int): The step between frequencies [Hz]
                
    Returns:
        (np.array): All the frequencies used in frequency sweep
    '''
    
    NO_FREQUENCIES = (end_freq - start_freq) // increment + 1

    return np.linspace(start_freq, end_freq, NO_FREQUENCIES)

def _get_no_in_bins(waves: np.array, no_frequencies: int) -> int:
    '''Gets the number of data points per frequency per sweep.
    
    Args:
        waves (np.array): 2D array of all the waves,
            shape #sweepsX#data-points-per-sweep
        no_frequencies (int): The number of frequencies sweeped through

    Returns
        (int): The number of datapoints per frequency per sweep
    '''
    return len(waves[0]) // no_frequencies


####################################################


class Setup:
    def __init__(self, exp_name: str, machine=None):
        '''Assembles the filenames in Drops w/o filetype endings (e.g. .h5)
        
        Args:
            exp_name (str): The experiment name. Should be
                CELL_NAME-XYZ-yyyy-mm-dd
            machine (str): Optional. The name of the computer running the
                resonance sweep. Corresponds to Drops location.
                Defaults to brix5
        '''

        if machine is None:
            machine = 'brix5'

        self.filename = f'/drops/data/{machine}/{exp_name}/{exp_name}'


##################################################


class PreProcess(Setup):
    '''Performs an intermediate conversion after an experiment to transform
    data from raw .sqlite3 (read in as a pd.DataFrame) to a .hdf5
    (written out as np.array). The latter doesn't need any preprocessing
    and is thus read in blazing fast (~10x performance improvement).
    '''
    
    def __init__(self, exp_name: str, machine=None):
        print('\tpreprocessing reso . . .\n')
        super().__init__(exp_name=exp_name, machine=machine)

    def _pull_data(self, n=None, table: str = 'table') -> pd.DataFrame:
        """Performs a sql query to pull data from Drops.
        
        Args:
            n (int): Optional. Skips every n-th row. Defaults to None
            table (str): Optional. The table name. Defaults to 'table'

        Returns:
            (pd.DataFrame): A dataframe of the raw data
        """

        db_location = f'{self.filename}.sqlite3'
        connection  = sqlite3.connect(db_location)

        query = f"SELECT * FROM '{table}'"
        if n is not None:
            query += f" WHERE _id % {n}=0"

        return pd.read_sql(sql=query, con=connection)

    def _save_datetimes(self, _id: pd.Series):
        """Saves datetime id-s in resodata to Drops.

        Args:
            _id (pd.Series): The timeseries column of the raw,
                original data.
        """

        _id.to_csv(f'{self.filename}_ids.csv')


    def _to_hdf(self, waves: np.array):
        """Writes raw data to an HDF5 file.
        
        Args:
            waves (np.array): The waveforms. See parse_waves()
        """

        with h5py.File(f'{self.filename}.h5', 'w') as f:
            f.create_dataset('waves', data=waves)

    def runall(self, n=None):
        df_raw = self._pull_data(n=n)
        self._save_datetimes(_id=df_raw['_id'])
        waves_array = _parse_waves(raw_data=df_raw)
        self._to_hdf(waves=waves_array)


##################################################3


class Resostat(Setup):
    '''Fetches Resostat data from drops and extract key info on experiment,
    including frequencies, no data points per bin and datetimes.

    Note: Only use after running PreProcess()
    '''
    
    def __init__(self, exp_name: str, machine=None):
        print('reading reso . . .\n')
        self.exp_name = exp_name
        super().__init__(exp_name=exp_name, machine=machine)

    def _get_waves(self, n=None) -> np.array:
        '''Loads into memory waveforms from Drops.
        
        Args:
            n (int): Subsample by selecting every n-th sweep

        Returns:
            (np.array): All waveforms
        '''

        while True:
            try:
                with h5py.File(f'{self.filename}.h5', 'r') as f:
                    waves = np.array(f['waves'][:], dtype=np.float16)
                break
            except FileNotFoundError:
                preprocess = PreProcess(exp_name=self.exp_name)
                preprocess.runall(n=n)            

        return waves if n is None else waves[::n]

    def _get_datetimes(self, n=None) -> pd.DataFrame:
        """Parses datetime id-s in resodata.

        Args:
            n (int): Subsample by selecting every n-th timestamp

        Returns:
           (pd.Series): The timestamps as pd.datetimes
        
        Note:
            This isn't optimal. Should be reading as a timeindex
            but it would ruin backwards compatibility.
            unit='s' b/c it's in unix epoch time, i.e.
                seconds elapsed since 1970-01-01
            utc=True to ensure no timezone f-ups when lining
                up timeseries w. Resostat
            As is it's probably going to fail when switching
            between daylight savings, but can't be bothered
            with for now.
        """

        ids = pd.read_csv(f'{self.filename}_ids.csv', index_col=0)
        utc = pd.to_datetime(ids['_id'], unit='s', utc=True)
        est = utc - pd.Timedelta('4 hours')

        return est if n is None else est.iloc[::n]

    
    def runall(self, exp_params: dict, n=None):
        waves = self._get_waves(n=n)
        frequencies_linspace = _get_frequencies(
            start_freq=exp_params['start_freq'],
            end_freq=exp_params['end_freq'],
            increment=exp_params['increment']
        )
        NO_IN_BIN = _get_no_in_bins(
            waves=waves,
            no_frequencies=len(frequencies_linspace)
        )
        datetimes = self._get_datetimes(n=n)
        
        return waves, frequencies_linspace, NO_IN_BIN, datetimes
