from datetime import datetime
import glob
import io
import itertools
import pandas as pd
import pytz

# For Biologic
usecols = ['time/s', 'cycle number', 'control/mA', 'Ewe/V']
raw_columns = ['cycle number', 'control/mA', 'Ewe/V']
proper_columns = ['cycle_no', 'I', 'V']

EST = pytz.timezone('US/Eastern')


class BioLogic:
    """Loads and parses propietary .mpt file from Biologic placed in Drops"""

    def __init__(self, exp_name, filename, machine='brix5'):
        print('reading biologic . . .\n')
        self.filename = f'{drop_pre}data/{machine}/{exp_name}/{filename}'

    def read_processed_file(self):
        return pd.read_csv(f'{self.filename}.csv', index_col=0, parse_dates=[0])

    def read_raw_file(self, skiprows: int = 75) -> pd.DataFrame:
        """Reads and parses raw potentiostat data from Drops.

        Args:
            skiprows (int): Optional. Number of rows to skip in .mpt file.
                It contains a heterogenous chunk of metadata. Defaults to 75
        
        Returns:
            (pd.DataFrame): Potentiostat data. Columns are
                timestamps (index), I and V
        
        Note:
            drop_pre is a prebuilt prefix in pithy
            utc=True ensures dt comparisons between different data sources
                (usually resostat) are (_almost_) idiot-proof.
        """
        
        def read_in(skiprows):
            """Don't touch"""
            return pd.read_csv(
                f'{self.filename}.mpt',
                sep='\t',
                skiprows=skiprows,
                usecols=usecols,
                index_col=0
            )
        
        # The number of rows to skip varies between files
        while True:
            try:
                bio_data = read_in(skiprows=skiprows)
            except ValueError:
                skiprows += 1
                continue
            break

        bio_data.index = pd.to_datetime(bio_data.index)
        bio_data.index = bio_data.index.tz_localize(pytz.utc).tz_convert(EST)

        mapper = dict(zip(raw_columns, proper_columns))
        
        return bio_data.rename(columns=mapper)

    def save(self, parsed_data: pd.DataFrame):
        """Saves parsed file for faster loading times.
        
        Args:
            parsed_data (pd.DataFrame): Potentiostat cycling data,
                columns datetimes (index), I and V
        """
        parsed_data.to_csv(f'{self.filename}.csv')

    def main(self):
        try:
            return self.read_processed_file()
        except FileNotFoundError:
            print('\tprocessed file doesn\'t exist. generating . . .\n')
            pass

        parsed_potentiostat_data = self.read_raw_file()
        self.save(parsed_data=parsed_potentiostat_data)
        
        return parsed_potentiostat_data


class Gamry:
    """LEGACY"""
    def __init__(self):
        # Read in all files in Drops folder
        self.files = glob.glob(f'{drop_pre}{DROPS_FOLDER}*')
        
    def read_data(self):
        '''An unwieldy thing. Maybe split? -> Si fractum non sit, noli id reficere'''
        list_of_file_contents_as_dfs = []
        filenames = []
        datestamps = []
        timestamps = []
        for j, file in enumerate(self.files):
            filenames.append(file.split('/')[-1])
            rawdata = open(file, encoding='latin-1').read()
            datestamp = rawdata.split('DATE\tLABEL\t')[1].split('\t')[0]
            datestamps.append(datestamp)
            timestamp = rawdata.split('TIME\tLABEL\t')[1].split('\t')[0]
            timestamps.append(timestamp)
            
            cvdata = rawdata.split('Compliance Voltage')[-1]
            cycles = cvdata.split('CURVE')[1:]
            for i in range(len(cycles)):
                df_temp = pd.read_csv(io.StringIO(cycles[i]),
                                      delim_whitespace=True,
                                      header=2,
                                      index_col=0)
                df_temp = df_temp.drop(df_temp.columns[3:], axis=1)
            list_of_file_contents_as_dfs.append(df_temp)

        dts_split = list(zip(datestamps, timestamps))
        dts_merged = [' '.join(tups) for tups in dts_split]
        dts_as_dt = pd.to_datetime(pd.Series(dts_merged),
                                   infer_datetime_format=True)

        return list_of_file_contents_as_dfs, dts_as_dt#, filenames
    
    @staticmethod
    def _datestr_to_datetime(dates_str: list):
        dates = []
        for date in dates_str:
            dates.append(datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))
        
        return dates
    
    @staticmethod
    def _filter_by_index(data, index):
        return list(itertools.compress(data, index))

    def filter_by_date(self, files, dts, date_limits: list):
        '''Filters files not part of experiement (by date)'''
        
        min_date, max_date = self._datestr_to_datetime(date_limits)
        is_file_relevant = [min_date <= dt <= max_date for dt in dts]
        
        relevant_data = self._filter_by_index(files, is_file_relevant)
        # reset_index is key -> see sort_chronologically
        dts_filtered = dts[is_file_relevant].reset_index(drop=True)

        return relevant_data, dts_filtered
    
    def sort_chronologically(self, files, dts):
        dts_sequential = dts.sort_values()
        index = dts_sequential.index.tolist()
        
        return [files[i] for i in index]
        
    def merge(self, files):
        merged = pd.concat(files, ignore_index=True)
        merged.fillna(0, inplace=True)
        return merged
        
    def extract_I_and_V(self, df):
        return df.A.to_numpy(), df.V.to_numpy()
        
    def get_timedelta(self, s):
        '''Generates a '''
        
        # 2 is arbitratily chosen as an SWEEP_INTERVAL between each file
        t_difference = [s[i] - s[i-1] if s[i] > s[i-1] else 2 for i in range(1, len(s))]
        t_difference.insert(0, 0)
        t_linspace = list(itertools.accumulate(t_difference))
        timedelta = pd.to_timedelta(t_linspace, unit='S')
        
        return timedelta
        
    def to_dataframe(self, I, V, timedelta):
        data = {'I': I, 'V': V}
        return pd.DataFrame(data=data, index=timedelta)
        
    def resample(self, df: pd.DataFrame, no_secs):
        '''Ensures timedeltas are equal for Resostat and potentiostat'''
        return df.resample(f'{no_secs}S').mean()
        
    def main(self):
        print('\nstarting w. potentiostat . . .')
        data, dts = self.read_data()
        relevant_data, relevant_dts = self.filter_by_date(data, dts, dates_conducted)
        chronological_data = self.sort_chronologically(relevant_data, relevant_dts)
        merged_data = self.merge(chronological_data)
        timedelta = self.get_timedelta(merged_data.s)
        I, V = self.extract_I_and_V(merged_data)
        gamry_df = self.to_dataframe(I, V, timedelta)
        df_resampled = self.resample(gamry_df, SWEEP_INTERVAL)
        print('. . . potentiostat data processing completed')
        print('\n------------------------\n')

        return df_resampled
        

   

