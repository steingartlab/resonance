from pithy3 import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import pathlib
from skimage import filters
import time


FIGSIZE = 10
DFT_COLORMAP = 'bone_r'


def save_fig(exp_name: str, fig_type: str, machine='brix5'):
    """Saves active figure to Drops.
    
    Args:
        exp_name (str): The experiment name. Should correspond
            to folder structure on Drops.
        fig_type (str): Figure name. For distinguishing between
            similar figures from same experiment.
        machine (str): Should correspond to folder structure on
            Drops.
    """

    img_name = f'/drops/data/{machine}/{exp_name}/{fig_type}_1.png'
    i = 1
    while True:
        i += 1
        path = pathlib.Path(img_name)
        if path.is_file():
            img_name = img_name.replace(f'{str(i-1)}.png', f'{str(i)}.png')
            continue
        plt.savefig(img_name, dpi=400, bbox_inches='tight', pad_inches=0)
        print(f'{fig_type} saved')
        break

def get_phase_change_indices(biologic):
    """Gets indices where current changes.
    
    Returns:
        (list): A boolean list of the indices where I switches phase
    """

    I = biologic.I.values
    I_shifted = biologic.I.shift(periods=1).values
    diff = I - I_shifted

    return [i for i, element in enumerate(diff) if abs(element)>3]

def plot_props(ax, n: int, gain: float):
    """Plots properties on subplot.
    
    'n' to denote if pcolormesh is being plotted at a lower res.
    'gain' for clarity
    
    Args:
        n (int): Indicates that figure was made with every n-th value
        gain (float): Experiment voltage gain
    """

    return ax.annotate(
        f'n={n}\ngain={gain}',
        xy=[0.91, 0.04],
        xycoords='axes fraction',
        color='k',
        backgroundcolor='w',
        alpha=0.5,
        fontsize=6
    )


class Figure:
    def __init__(self, dpi):
        clf()

        if dpi is None:
            dpi=200

        self.dpi=dpi


class FFTPlot(Figure):
    def __init__(self, exp_params: dict, freq_bins: np.array, amps: np.array, datetimes: pd.Series, potentiostat: pd.DataFrame, waves: pd.DataFrame,  pca_components: list, temperature: pd.DataFrame = None,pressure: pd.DataFrame = None,n=None, dpi=None):
        """Plots FFT, amplitude, and potentiostat on same figure.
        
        Args:
            exp_params (dict): Same as on top of every experiment module
            freq_bins (1D np.array): Frequency bin centers, resultant from DFT
            amps (2D np.array): FFT powers
            datetimes (pd.Series): Parsed resostat datetimes (time zone aware: EST)
            potentiostat (pd.DataFrame): Potentiostat cycling data: timestamp, V & I
            waves (pd.DataFrame): Total wave amplitudes, shape [no_sweeps, no_freqs]
            temperature (pd.DataFrame): Temperature data: timestamp, internal & battery
            n: Optional. To skip every n-th value. Allows for significant performance
                improvements, at cost of image resolution. Defaults to None
            dpi: Optional. Image quality. Defaults to 200.
         """

        print('plotting DFT . . .\n')
        super().__init__(dpi=dpi)

        self.NO_SUBPLOTS = 5
        self.fig, self.axs = plt.subplots(
            self.NO_SUBPLOTS,
            1,
            figsize=(6,11),
            gridspec_kw={'height_ratios': [2, 1, 1, 1, 1]}
        )
        self.FFT_ROW = 0
        self.PCA_ROW = 1
        self.AMP_ROW = 2
        self.T_ROW = 3
        self.P_ROW = 3
        self.POT_ROW = 4
        
        self.n = n
        self.exp_params = exp_params
        self.freq_bins = freq_bins
        self.amps = amps
        self.datetimes = datetimes
        self.potentiostat = potentiostat
        self.waves = waves
        self.temperature = temperature
        self.pressure = pressure
        self.pca_components = pca_components

        # Speeds up plotting
        if self.n is not None:
            self.freq_bins = self.freq_bins[::n]
            self.amps = self.amps[:, ::n]

    def _plot_fft(self, original_frequencies):
        self.axs[self.FFT_ROW].pcolormesh(
            self.datetimes.values,
            self.freq_bins/1000,
            self.amps.T,
            cmap=DFT_COLORMAP,
            shading='auto'
        )

        # self.axs[self.FFT_ROW].set_ylim([250, 330])  # Manual
        self.axs[self.FFT_ROW].set_ylim(
            [min(original_frequencies)/1000, 
             max(original_frequencies)/1000+1]
        )  # Automatic
        self.axs[self.FFT_ROW].set_ylabel('Frequency [kHz]')

    def _plot_pca(self, i=0):
        self.axs[self.PCA_ROW].plot(
            self.datetimes.values[1:],
            smooth(self.pca_components, 4),
            c='k'
        )

        self.axs[self.PCA_ROW].set_ylim([min(self.pca_components), max(self.pca_components)])
        # self.axs[self.PCA_ROW].set_ylim([round(min(self.pca_components[0]),2)-0.01, round(max(self.pca_components[0]),2)+0.01])
        self.axs[self.PCA_ROW].set_ylabel('Total Frequency\nAmplitude [a.u.]')

    def _plot_original_freqs(self, original_frequencies):
        """Note: Only use for a low number of frequencies,
        otherwise it clutters the fft plot"""

        self.axs[self.FFT_ROW].hlines(
            y=original_frequencies/1000,
            xmin=min(self.potentiostat.index),
            xmax=max(self.potentiostat.index),
            colors='k',
            linewidth=0.5
        )

    def _plot_potentiostat(self):
        self.axs[self.POT_ROW].plot(self.potentiostat.index, self.potentiostat.V, '#0033a0')

        self.ax2 = self.axs[self.POT_ROW].twinx()
        self.ax2.plot(self.potentiostat.index, self.potentiostat.I/1000, 'gray')

        # Properties
        self.axs[self.POT_ROW].set_ylabel('Voltage [V]', c='#0033a0')
        self.ax2.set_ylabel('Current [A]', c='gray') 
        self.axs[self.POT_ROW].set_zorder(1)
        self.axs[self.POT_ROW].set_frame_on(False)
        self.axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        self.axs[-1].set_xlabel('Time [h]')

    def _plot_resostat(self, frequency_index=None):
        if frequency_index is None:
            frequency_index = -1
        curve = self.waves.iloc[:, frequency_index]
        curve_normalized = (curve - min(curve)) / (max(curve)-min(curve))
        self.indices = self.waves.index <= max(self.potentiostat.index)
        frequency = round(float(self.waves.columns[frequency_index]) / 1000., 2)
        self.axs[self.AMP_ROW].plot(
            self.waves.index[self.indices],
            curve_normalized[self.indices],
            c='k',
            label=f"{frequency} kHz"
        )

        # Properties
        self.axs[self.AMP_ROW].set_ylim([0, 1.1])
        self.axs[self.AMP_ROW].set_ylabel('Normalized\nAmplitude [a.u.]')
        self.axs[self.AMP_ROW].legend(
            loc='upper left', 
            framealpha=.5,
            fontsize='x-small'
        )

    def _fill(self, vlines, vline_index, color):
        for i in range(1, self.NO_SUBPLOTS-1):
            self.axs[i].fill_betweenx(
                np.linspace(0, max(self.pca_components), 100),
                self.potentiostat.index[vlines[vline_index]],
                self.potentiostat.index[vlines[vline_index+1]],
                color=color,
                alpha=.05
            )

    def _plot_vlines(self, max_freq):
        vlines_index = get_phase_change_indices(self.potentiostat)
        # My oh my, that's crude
        color = 'red'
        for i in range(0, len(vlines_index)-1):
            if i % 2:
                continue
            color = 'green' if color == 'red' else 'red'
            self._fill(vlines=vlines_index, vline_index=i, color=color)

        # for i in range(self.NO_SUBPLOTS-1):
        self.axs[0].vlines(
            x=self.potentiostat.index[vlines_index],
            ymin=0,
            ymax=max(self.pca_components),
            colors='gray',
            linestyles='dashed',
            linewidth=0.5,
            zorder=3
        )
        
    def _plot_colorbar(self):
        sm = plt.cm.ScalarMappable(cmap=DFT_COLORMAP,
                                   norm=plt.Normalize(vmin=0,
                                                      vmax=1))
        # left, bottom, width, height        
        cbar_ax = self.fig.add_axes([0.91, 0.76, 0.012, 0.10])
        cbar_ticks = np.linspace(0, 1, 3)
        cb = self.fig.colorbar(sm, cax=cbar_ax, ticks=cbar_ticks)
        cbar_ax.set_yticklabels([str(tick) for tick in cbar_ticks], fontsize=7)
        cb.set_label(label='Normalized\nPower [a.u.]', size=7)

    def _plot_temperature(self, is_smooth=True):        
        colormap = plt.cm.binary(np.linspace(0, 1, 1 + len(self.temperature.columns)))
        for i, sensor in enumerate(self.temperature):
            x = self.temperature.index[1:]
            y = smooth(self.temperature[sensor], 180)
            self.axs[self.T_ROW].plot(
                x - pd.Timedelta('4 hours'),
                y,
                c=colormap[i+1],
                label=sensor
            )

        # Properties
        self.axs[self.T_ROW].set_ylabel('Temperature [°C]')
        self.axs[self.T_ROW].set_ylim([20, 30])
        # self.axs[self.T_ROW].set_ylim(
        #     [round(min(self.temperature)/5,2)*5,
        #      round(max(self.temperature)/5,2)*5]
        # )
        self.axs[self.T_ROW].legend(
            loc='upper left',
            framealpha=.5,
            fontsize='x-small'
        )

    def _plot_pressure(self):
        self.p_ax = self.axs[self.T_ROW].twinx()
        self.pressure['voltage'] = self.pressure['voltage'] / 1000

        p = smooth(self.pressure['voltage'], 60)  # Is actually pressure, but need for backwards compatibility     
        self.p_ax.plot(
            self.pressure.index[1:] - pd.Timedelta('4 hours'),
            p,
            c='#0033a0'
        )

        # Properties
        self.p_ax.set_ylabel('Pressure [kPa]', c='#0033a0')
        try:
            self.p_ax.set_ylim([min(self.pressure.voltage)-0.5, max(self.pressure.voltage)+0.5])
        except AttributeError:
            self.p_ax.set_ylim([min(self.pressure.pressure)-0.5, max(self.pressure.pressure)+0.5])

    def _set_properties(self, original_frequencies):
        # Set axes limits
        for i in range(self.NO_SUBPLOTS):
            self.axs[i].set_xlim(
                [min(self.potentiostat.index),
                 max(self.potentiostat.index)]
            )
            if i != self.NO_SUBPLOTS-1:
                self.axs[i].get_xaxis().set_visible(False)

        # Misc
        self.fig.suptitle(f"{self.exp_params['exp_name']}. {self.exp_params['c_rate']} w/o casing")
        plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.94)
        self.axs[0].set_title(f"Frequency sweep {round(min(original_frequencies)/1000)}-{round(max(original_frequencies)/1000)} kHz. Interval {round((original_frequencies[1]-original_frequencies[0])/1000, 2)} kHz", fontsize=10)


    def main(self, original_frequencies, gain=None, frequency_index=None, is_smooth=True):
        self._plot_fft(original_frequencies=original_frequencies)
        plot_props(ax=self.axs[self.FFT_ROW], n=self.n, gain=gain)
        self._plot_potentiostat()
        self._plot_pca()
        self._plot_resostat(frequency_index=frequency_index)
        self._plot_vlines(max_freq=max(original_frequencies))
        self._plot_colorbar()

        if self.temperature is not None:
            self._plot_temperature(is_smooth=is_smooth)
        if self.pressure is not None:
            self._plot_pressure()

        self._set_properties(original_frequencies=original_frequencies)
        # self.plot_original_freqs(original_frequencies=original_frequencies)

        save_fig(exp_name=self.exp_params['exp_name'], fig_type='FFT_subplots')

        showme(dpi=200)
        clf()


#######################################


class TilePlot(Figure):
    def __init__(self, title: str, nrows, ncols, dpi=None):
        super().__init__(dpi=dpi)

        height_ratios = list(np.ones(self.nrows))
        height_ratios.append(self.nrows//self.ncols+1)
        self.fig, self.axs = plt.subplots(
            nrows=self.nrows,
            ncols=self.ncols,
            gridspec_kw={'height_ratios': height_ratios},
            figsize=(FIGSIZE, FIGSIZE)
        )
        colormap = plt.cm.coolwarm(np.linspace(0, 1, len(df.columns)))
        self.fig.suptitle(title)
        self.fig.supylabel('Normalized Amplitude [a.u.]', x=0.08, y=0.7, fontsize=14)
        self.fig.supxlabel('t [h]', fontsize=20)

    def plot_waveforms(self):
        i = 0
        for row in range(NROWS):
            for col in range(NCOLS):
                curve = df.iloc[:, i]
                curve_normalized = (curve - min(curve)) / (max(curve)-min(curve))
                self.axs[row, col].plot(df.index, curve_normalized.values, c=colormap[i])
                self.axs[row, col].axis('off')
                i += 1

    def plot_colorbar(self):
        frequencies = list(df.columns)
        sm = plt.cm.ScalarMappable(cmap='coolwarm',
                                norm=plt.Normalize(vmin=min(frequencies)/1000,
                                                    vmax=max(frequencies)/1000))
        self.fig.subplots_adjust(right=0.8)
        cbar_ax = self.fig.add_axes([0.85, 0.35, 0.02, 0.5])
        cbar_ticks = np.linspace(min(frequencies), max(frequencies), 10)//1000
        self.fig.colorbar(sm, cax=cbar_ax, label='Frequencies [kHz]', ticks=cbar_ticks)
        cbar_ax.set_yticklabels([str(int(tick)) for tick in cbar_ticks])

    def main(self, df: pd.DataFrame, datetimes):
        self.plot_waveforms()
        self.plot_potentiostat()
        self.plot_colorbar()
        
        # fig.supxlabel('SoC  [%]', fontsize=16)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        showme(dpi=400)
        clf()

########################################################33


class GhostPlot(Figure):
    """Stacks the cycling data to gauge reproducibility."""
    def __init__(self, exp_params: dict, waves: pd.DataFrame, biologic: pd.DataFrame, dpi=None):
        super().__init__(dpi=dpi)

        self.exp_params = exp_params
        self.waves = waves
        self.biologic = biologic
        
        self.fig, self.axs = plt.subplots()

    def get_amp_indices(self, cycle_no, indices):
        """Gets where the Reso-amplitude is in cycle no x.
        
        Args:
            cycle_no (int): cycle number
        
        Returns:
            is_in_cycle (list): A boolean list of the indices
                where amplitude is in a particular cycle
        """

        if cycle_no == self.exp_params['no_cycles']:
            is_in_cycle = self.biologic.index[indices[-3]] < self.waves.index
        elif cycle_no > 1:
            index = 4 * cycle_no
            # print(indices, index, len(self.biologic.index))
            upper = self.biologic.index[indices[index-1]] < self.waves.index
            # print(len(self.biologic), len(indices), index)
            lower = self.waves.index < self.biologic.index[indices[index+3]]
            is_in_cycle = [l and u for l, u in zip(lower, upper)]
        elif cycle_no == 1:
            is_in_cycle = self.waves.index < self.biologic.index[indices[3]]
        else:
            pass

        return is_in_cycle

    def normalize_amps(self, frequency_index: int):
        amps = self.waves.iloc[:, frequency_index]
        normalized_amps = (amps - min(amps)) / (max(amps)-min(amps))

        return normalized_amps

    def plot_amplitude_curves(self, indices, no_cycles: int, amps):
        
        colormap = plt.cm.coolwarm(np.linspace(0, 1, no_cycles))

        for cycle_no in range(1, no_cycles+1):
            amp_indices = self.get_amp_indices(
                indices=indices, 
                cycle_no=cycle_no
            )
            t = np.linspace(0, 1, np.count_nonzero(amp_indices))
            # These two atrocious if statements are manual for 2022-02-18 experiment
            # if cycle_no==1:
            #     t = np.linspace(0.37, 1, np.count_nonzero(amp_indices))
            # if cycle_no == 5:
            #     amps_last_cycle = amps[amp_indices]
            #     amps_last_cycle = amps_last_cycle[:186]
            #     t = np.linspace(0.405, 1, len(amps_last_cycle))
            #     self.axs.plot(
            #         t,
            #         amps_last_cycle,
            #         c=colormap[cycle_no-1],
            #         linewidth=1.5,
            #         label=cycle_no
            #     )
            # else:
            self.axs.plot(
                t,
                amps[amp_indices],
                c=colormap[cycle_no-1],
                linewidth=1.5,
                label=cycle_no
            )

    def plot_differences(self, amps, indices, no_cycles):
        cycle_indices = list()
        for cycle_no in range(2, no_cycles):
            cycle_indices.append(self.get_amp_indices(
                indices=indices, 
                cycle_no=cycle_no
                )
            )
        base_cycle = amps[cycle_indices[0]].values
        second_cycle = amps[cycle_indices[1]].values
        print(len(base_cycle), len(second_cycle))
        differences = second_cycle[:-7] - base_cycle
        t = np.linspace(0, 1, len(differences))
        self.axs[1].plot(t, differences, 'k')
        third_cycle = amps[cycle_indices[2]].values
        print(len(third_cycle))
        differences = third_cycle[:-6] - base_cycle
        t = np.linspace(0, 1, len(differences))
        self.axs[1].plot(t, differences, 'k')


    def _set_properties(self):
        self.axs.set_ylabel('Normalized\nAmplitude [a.u.]')
        self.axs.title.set_text(f"{self.exp_params['exp_name']}. {self.exp_params['c_rate']} w/o casing")
        self.axs.legend(title='cycle number', loc='upper left', framealpha=.5)
        # self.axs[0].get_xaxis().set_visible(False)

        self.axs.set_xlabel('Normalized time [a.u.]')

        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        showme(dpi=self.dpi)
        clf()

    def main(self, frequency_index: int = -2):
        NO_CYCLES = self.exp_params['no_cycles']
        indices = get_phase_change_indices(self.biologic)
        normalized_amps = self.normalize_amps(frequency_index=frequency_index)
        self.plot_amplitude_curves(
            indices=indices,
            no_cycles=NO_CYCLES,
            amps=normalized_amps
        )
        # self.plot_differences(amps=normalized_amps, indices=indices, no_cycles=NO_CYCLES)
        self._set_properties()

        # Shaded areas
        # color = 'red'
        # for i in range(0, 4):
        #     if i % 2:
        #         continue
        #     color = 'green' if color == 'red' else 'red'
        #     fill(vline_index=i, color=color)
        # ax.vlines(
        #     x=t[amp_indices],
        #     ymin=0,
        #     ymax=1,
        #     colors='gray',
        #     linestyles='dashed',
        #     linewidth=0.5
        # )



class SingleFrequencyPlot(Figure):

    def __init__(self, waves: pd.DataFrame, biologic: pd.DataFrame, exp_params: dict, dpi=None):
        """Plots a single frequency band with the potenetistat data
 
 
       Args:
            waves (pd.DataFrame): The waveforms.
                See GT_Resonance_Signal_Processing.Signal_Processing for more info
            biologic (pd.DataFrame): The cycling data. Has index of datetime64ns
                w. columns V and I
            c_rate (str): The experiment's C-rate, e.g. '1C' or 'C/4'
            frequency_index (int): The index of the frequency to plot.
                See waves.columns for more info.
        
        """

        super().__init__(dpi=dpi)
        self.exp_params = exp_params
        self.waves = waves
        self.biologic = biologic

        self.fig, self.axs = plt.subplots(2, 1, figsize=(5, 6)) 
        self.axs[0].set_ylabel('Normalized\nAmplitude [a.u.]')
        self.axs[0].title.set_text(f"{self.exp_params['exp_name']}. {self.exp_params['c_rate']} w/o casing")

    def _fill(self, vlines, vline_index, color, ax_index=0):
        self.axs[ax_index].fill_betweenx(
            np.linspace(0, 1, 10),
            self.biologic.index[vlines[vline_index]],
            self.biologic.index[vlines[vline_index+1]],
            color=color,
            alpha=.1
        )

    def _plot_resostat(self, frequency_index):
        curve = self.waves.iloc[:, frequency_index]
        curve_normalized = (curve - min(curve)) / (max(curve)-min(curve))
        self.indices = self.waves.index <= max(self.biologic.index)
        self.axs[0].plot(
            self.waves.index[self.indices],
            curve_normalized[self.indices],
            c='k',
            label=f"141 kHz"
        )

    def _plot_shaded_areas(self):
        # Shaded areas
        vlines = get_phase_change_indices(self.biologic)
        color = 'red'
        for i in range(0, len(vlines)-1):
            if i % 2:
                continue
            color = 'green' if color == 'red' else 'red'
            self._fill(vlines=vlines, vline_index=i, color=color)
        self.axs[0].vlines(
            x=self.biologic.index[vlines],
            ymin=0,
            ymax=1,
            colors='gray',
            linestyles='dashed',
            linewidth=0.5
        )
    
    def _plot_potentiostat(self):
        ax2 = self.axs[1].twinx()
        self.axs[1].plot(self.biologic.index, self.biologic.V, 'r')
        ax2.plot(self.biologic.index, self.biologic.I/1000, 'b')
        self.axs[1].set_ylabel('Voltage [V]', c='r')
        ax2.set_ylabel('Current [A]', c='b')   

    def _set_properties(self):
        self.axs[0].get_xaxis().set_visible(False)
        self.axs[0].legend(loc='upper right', framealpha=0.5)
        timedelta = pd.Timedelta('30 minutes')
        for i in range(2):
            self.axs[i].set_xlim(
                [min(self.waves.index)-timedelta,
                self.waves.index[np.count_nonzero(self.indices)-1]+timedelta]
            )
        self.axs[1].set_xlabel('t [h]')
        self.axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        self.axs[1].set_zorder(1)
        self.axs[1].set_frame_on(False)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

    def main(self, frequency_index=-2):
        self._plot_resostat(frequency_index=frequency_index)
        self._plot_shaded_areas()
        self._plot_potentiostat()

        self._set_properties()
        showme(dpi=self.dpi)
        clf()

################################################


class Spectrogram(Figure):
    """A spectrogram with dashed lines perpendicular to time-axis, identifying potentiostat phase change."""

    def __init__(self, n=None, dpi=None):
        super().__init__(dpi=dpi)
        self.n = n
        self.fig, self.ax = plt.subplots()

    def _set_properties(self, exp_params):
        self.ax.set_ylim([exp_params['start_freq']/1000, exp_params['end_freq']/1000+1])
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        self.ax.set_xlabel('t [h]')
        self.ax.set_ylabel('Frequency [kHz]')
        self.ax.set_title(exp_params['exp_name'])

    
    def _spectrogram(self, datetimes, freq_bins, amps):
        self.ax.pcolormesh(
            datetimes.values,
            freq_bins/1000,
            amps.T,
            cmap='coolwarm',
            shading='auto'
        )
    
    def _binarygram(self, datetimes, freq_bins, amps):
        sobel_filtered = filters.sobel(amps)

        self.ax.pcolormesh(
            datetimes.values,
            freq_bins/1000,
            sobel_filtered.T,
            cmap='gray',
            shading='auto'
        )

    def _vlines(self, potentiostat, max_freq):
        vlines_index = get_phase_change_indices(potentiostat)

        self.ax.vlines(
            x=potentiostat.index[vlines_index],
            ymin=0,
            ymax=max_freq/1000,
            colors='blue',
            linestyles='dashed',
            linewidth=0.5,
            zorder=3
        )
    
    def plot(self, datetimes, freq_bins, amps, potentiostat, exp_params, is_filtered: bool = True):

        if self.n is not None:
            amps = amps[:, ::self.n]
            freq_bins = freq_bins[::self.n]
            plot_props(ax=self.ax, n=self.n)

        if is_filtered:
            self._binarygram(
                datetimes=datetimes,
                freq_bins=freq_bins,
                amps=amps
            )
        else:
            self._spectrogram(
                datetimes=datetimes,
                freq_bins=freq_bins,
                amps=amps
            )
        
        self._vlines(
            potentiostat=potentiostat,
            max_freq=exp_params['end_freq']
        )

        self._set_properties(exp_params)
        
        fig_type = 'binarygram' if is_filtered else 'spectrogram'
        save_fig(exp_name=exp_params['exp_name'], fig_type=fig_type)
        showme(dpi=self.dpi)
        clf()