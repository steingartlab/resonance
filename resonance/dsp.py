import h5py
import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats

import utils


def _get_first_peaks(waves: np.array) -> np.array:
    '''Gets the first peak in each waveform for each frequency for each sweep.

    Analogous to get_total_amplitudes()

    Args:
        waves (np.array): 2D array of all waveforms.
    
    Returns:
        (np.array): Array with first peak from each wave
    '''
    
    first_peaks = np.zeros((waves.shape[0], waves.shape[1]))
    for i in range(waves.shape[0]):
        for j in range(waves.shape[1]):
            peaks_indices = signal.find_peaks(waves[i, j, :], height=0.001)
            # Das ist ein Problem - why don't all the waveforms have peaks?
            if len(peaks_indices[0])==0:
                first_peaks[i, j] = 0
            else:
                first_peak_index = peaks_indices[0][0]
                first_peaks[i, j] = waves[i, j, first_peak_index]
                
    return first_peaks

def remove_outliers(df: pd.DataFrame, z_max: int = 3) -> pd.DataFrame:
    '''Removes outliers in each column in dataframe.
    
    Does so by calculating the z-score, i.e. the number of stddevs a
    value is from the mean.

    Args:
        df (pd.DataFrame): Any dataframe with potential outliers.
        z_max (int): Optional. The maximum number of stdevs to allow.
            Defaults to 3.
    
    Returns:
        (pd.DataFrame): The outlier-free waveforms
    '''
    
    z = np.abs(stats.zscore(df))
    
    return df[(z < z_max).all(axis=1)]

def _smooth_amplitudes(df: pd.DataFrame, window_len: int) -> pd.DataFrame:
    """Smooths amplitude curves using Dan's smoothing-function.
    
    Args:
        df (pd.DataFrame): 
        window_len (int): Window length, i.e. how many values to include in
            each smoothing call.

    Returns:
        smoothed (pd.DataFrame): Smoothed wave amplitudes.
    """

    smoothed = df.copy(deep=True)
    smoothed.drop(smoothed.tail(1).index, inplace=True)
    for j in range(len(df.columns)):
        smoothed.iloc[:, j] = utils.smooth(
            df.iloc[:, j],
            window_len=window_len
        )

    return smoothed

def _get_sample_rate(no_points_per_sweep: int, no_frequencies: int, dwell: float) -> float:
    """Calculates sampling rate in ResoStat frequency sweep.

    Args:
        no_points_per_sweep (int): The number of data points collected
            in each sweep
        no_frequencies (int): The total number of frequencies used
            in experiment
        dwell (float): The time spent on each frequency during each sweep

    Returns:
        (float): The sampling rate [s]
    """
    return no_points_per_sweep / no_frequencies / dwell

def _is_sample_rate_adequate(sample_rate: float, max_frequency: int) -> None:
    """Sanity check for FFT quality.
    
    Sample_rate must be at least twice the highest signal frequency.

    Args:
        sample_rate (float): The calculated sample rate,
            see _get_sample_rate()
        max_frequency (int): The highest frequency used in frequency sweep
    """

    try:
        assert sample_rate >= 2 * max_frequency
    except AssertionError:
        print("\nSample rate doesn\'t fulfill requirements.\n")

def _zero_pad(waves: np.array, pad_x: int = 2) -> np.array:
    """Zero pads waves to increase FFT frequency resolution.

    Standard practice for FFTs. See e.g. https://bit.ly/3BPcG4q.

    Args:
        waves (np.array): Original 2D wave matrix.
            Shape [no_sweeps, no_points_in_sweep]
        pad_x (int): Optional. The padding multiplication factor.
            Defaults to 2.
    """

    no_points_per_sweep_padded = len(waves[0]) * pad_x
    no_waves = len(waves)
    z = np.zeros((no_waves, no_points_per_sweep_padded), dtype='uint8')
    padded_waves = np.concatenate((z, waves, z), axis=1, dtype=np.float16)

    return padded_waves

def _detrend(waves: np.array) -> np.array:
    """Hmm

    Args:
        waves (np.array): Zero-padded waves

    Returns:
        (np.array): Detrended waves
    
    Note:
        Must be run after _zero_pad()
    """
    return signal.detrend(waves)

def _get_absolute_transform(fft: np.array) -> np.array:
    """Calculates total length of transform for complex number.

    The FFT results in complex-valued numbers.
    np.abs() evaluates sqrt(a²+b²) for complex number a+bi.

    Helper function for _get_fft()

    Args:
        fft (np.array): The raw results from FFT

    Returns:
        (np.array): Length of every complex-valued number in FFT
    """
    return np.abs(fft)

def _normalize_amps(amps: np.array) -> np.array:
    """Normalizes to simplify plotting.

    Helper function for _get_fft()
    
    Args:
        amps(np.array): Frequency amplitudes from FFT

    Note:
        Run after _get_absolute_transform()
    """
    return np.divide(amps.T, np.max(amps, axis=-1)).T

def _get_fft(waves: np.array, sample_rate: float) -> [np.array, np.array]:
    """Calculates FFT.
    
    Args:
        waves (np.array): 2D wave array of waveforms,
            shape [no_sweeps, no_points_in_sweep]
        sample_rate (float): see _get_sample_rate()

    Returns:
        freq_bins (np.array): 1D array of frequency bin centers
        normalized_amps (np.array): 2D array of frequency powers,
            shape [no_sweeps, no_freq_bins]
    """

    amps_complex = np.fft.rfft(waves)
    amps_length = _get_absolute_transform(fft=amps_complex)
    normalized_amps = _normalize_amps(amps=amps_length)

    window_len = len(waves[0])
    freq_bins = np.fft.rfftfreq(n=window_len, d=1/sample_rate)

    return freq_bins, normalized_amps

def _subsample_waves(waves: np.array, n: int) -> np.array:
    return waves[::n, :]


class SignalProcessing:
    '''General DSP acrobatics to parse Resodata.

    Attributes:
        frequencies (list): The frequencies used in sweep.
        no_in_bin (int): Number of samples per frequency
            per sweep.
    '''
    
    def __init__(self, frequencies: list, no_in_bin: int):
        print('processing signal . . .\n')
        self.frequencies = frequencies
        self.no_in_bin = no_in_bin
        
    def _group_by_frequency(self, waves: np.array) -> np.array:
        '''Groups all waves into a 3D array w. shape of:
                no_sweeps: len(waves)
                no_frequencies: (len(frequencies))
                no of data points per freq per sweep (no_in_bin)
                
        Returns:
            (np.array): 3D wave matrix
        '''

        def get_index(freq_no):
            return self.no_in_bin * freq_no

        waves_matrix = np.zeros((len(waves),
                                 len(self.frequencies),
                                 self.no_in_bin),
                                dtype=np.float16)
        for i, wave in enumerate(waves):
            for frequency_no in range(len(self.frequencies)):
                j_lower = get_index(freq_no=frequency_no)
                j_upper = get_index(freq_no=1+frequency_no)
                waves_matrix[i, frequency_no, :] = wave[j_lower:j_upper]
                
        return waves_matrix

    def _get_total_amplitudes(self, waves: np.array, datetimes: pd.Series) -> pd.DataFrame:
        '''Gets total waveform amplitude for each frequency for each sweep

        Args:
            waves (np.array): 2D array of all waveforms
            datetimes (pd.Series): parsed timestamps of each frequency sweep.
        
        Returns:
            (pd.DataFrame): Total amplitude for each frequency in each sweep.
        '''

        total_A = np.zeros((waves.shape[0], waves.shape[1]))
        for i in range(waves.shape[0]):
            for j in range(waves.shape[1]):
                waveform = waves[i, j, :]
                total_A[i, j] = np.sum(np.dot(waveform, waveform))
                
        total_A_df = pd.DataFrame(data=total_A,
                                  columns=self.frequencies,
                                  index=datetimes)

        return total_A_df

    def runall(self, waves, datetimes, window_len=10):
        waves_3D_array = self._group_by_frequency(waves)
        total_A = self._get_total_amplitudes(waves_3D_array, datetimes)
        total_A_wo_outliers = remove_outliers(total_A)
        total_A_wo_outliers_smoothed = _smooth_amplitudes(total_A_wo_outliers, window_len)
        
        return total_A_wo_outliers_smoothed


class FFT:
    """Rehashed FFT from Rob Mohr.
    
    Attributes:
        filename (str): The Drops filename

    Usage:
        (automated w. FFT.main())

    """

    def __init__(self, exp_name):
        print('reading DFT . . .\n')
        machine = 'brix5'
        self.filename = f'/drops/data/{machine}/{exp_name}/{exp_name}_FFT.h5'

    def _save(self, freq_bins: np.array, amps: np.array):
        """Writes FFT to an HDF5 file.
        
        Args:
            freq_bins (np.array): 1D array of frequency bin centers
            normalized_amps (np.array): 2D array of frequency powers,
                shape [no_sweeps, no_freq_bins]
        """

        with h5py.File(self.filename, 'w') as f:
            f.create_dataset('amps', data=amps)
            f.create_dataset('freq_bins', data=freq_bins)

    def _read(self) -> [np.array, np.array]:
        """Reads in previously saved FFT.

        Returns:
            freq_bins (np.array): Frequency bin centers
            amps (np.array): FFT powers
        
        Note:
            I'm purposely not doing n=n here where
            every n-th sweep is read. I do it instead when
            plotting to ensure compliance with datetimes
        """

        with h5py.File(self.filename, 'r') as f:
            # Note to self: don't change dtype. plt doesn't like it
            amps = np.array(f['amps'][:], dtype=np.float32)
            freq_bins = np.array(f['freq_bins'], dtype=np.float32)

        return freq_bins, amps
    
    def main(self, waves: np.array, frequencies: list, dwell: float, pad_x=2, n=None):
        try:
            return self._read()
        except FileNotFoundError:
            print('\tnvm, file not found -> generating FFT . . .\n')
        
        SAMPLE_RATE = _get_sample_rate(
            no_points_per_sweep=waves.shape[-1],
            no_frequencies=len(frequencies),
            dwell=dwell
        )
        _is_sample_rate_adequate(
            sample_rate=SAMPLE_RATE,
            max_frequency=max(frequencies)
        )

        padded_waves = _zero_pad(waves=waves, pad_x=pad_x)
        detrended_waves = _detrend(waves=padded_waves)

        if n is not None:
            detrended_waves = _subsample_waves(
                waves=detrended_waves,
                n=n
            )

        freq_bins, amps = _get_fft(
            waves=detrended_waves, 
            sample_rate=SAMPLE_RATE
        )

        self._save(
            freq_bins=freq_bins,
            amps=amps
        )

        return freq_bins, amps
