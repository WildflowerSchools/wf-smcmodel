"""
Implement databases in memory for reading and writing data while performing
sequential Monte Carlo (SMC) simulation and inference.
"""

import smcmodel.shared_constants
from . import Database
from . import DataQueue
import datetime_conversion
import numpy as np

class DatabaseMemory(Database):
    """
    Implement database in memory for reading and writing data based on timestamp.
    """

    def __init__(self, structure, num_samples):
        """
        Constructor for DatabaseMemory class.

        Initializes an empty database with the specified variable structure and number of samples per timestamp.

        The keys of structure should be the variable names associated with the
        object that the database is storing (e.g., if the database is storing
        data for the state of the system, keys might be 'positions' and
        'velocities'). The values should be dicts containing the structure info
        for that those variables: 'shape' and 'type' (see examples of implemented
        SMCModel class for examples of this structure).

        Parameters:
            structure (dict): Structure of the object that the database is storing
            num_samples (int): Number of samples per timestamp
        """
        self.structure = structure
        self.num_samples = num_samples
        self._timestamp_list = []
        self._data_dict_list = []

    def write_data(self, timestamp, single_time_data):
        """
        Write data to database with a timestamp.

        The timestamp should be a Python datetime object (timezone-aware or
        timezone-naive), a Numpy datetime64 object, a Pandas Timestamp object
        (timezone-aware or timezone-naive), a string in ISO format
        (timezone-aware or timezone-naive), or seconds since the Unix epoch
        (float or int).

        The keys of single_time_data should be the variable names associated
        with the object that the database is storing (e.g., if the database is
        storing data for the state of the system, keys might be 'positions' and
        'velocities'). The values should be array-like (convertible to Numpy
        arrays) with shape (number of samples at each timestep, [shape of
        variable]).

        Parameters:
            timestamp (datetime): Timestamp associated with the data
            single_time_data (dict): Data for that timestamp
        """
        timestamp = datetime_conversion.to_posix_timestamp(timestamp)
        if set(single_time_data.keys()) != set(self.structure.keys()):
            raise ValueError('Keys in new data don\'t match variable structure of database')
        for variable_name in self.structure.keys():
            single_time_data[variable_name] = np.asarray(single_time_data[variable_name])
            array_num_samples = single_time_data[variable_name].shape[0]
            if array_num_samples != self.num_samples:
                raise ValueError('Database expects {} samples but new data for {} appears to contain {} samples'.format(
                    self.num_samples,
                    variable_name,
                    array_num_samples
                ))
        self._timestamp_list.append(timestamp)
        self._data_dict_list.append(single_time_data)

    def fetch_data(self, start_datetime = None, end_datetime = None):
        """
        Fetch time series data for a specified time span.

        The datetime arguments should be Python datetime objects (timezone-aware
        or timezone-naive), Numpy datetime64 objects, Pandas Timestamp objects
        (timezone-aware or timezone-naive), strings in ISO format
        (timezone-aware or timezone-naive), or seconds since the Unix epoch
        (float or int).

        If start_datetime is not specified, returned data starts with the
        earliest data in the database. If end_datetime is not specified,
        returned data ends with the latest data in the database.

        The returned timestamps and time series data will be in time order
        regardless of the order in which the data was written to the database.

        The keys of the returned time series data are the variable names
        associated with the object that the database is holding (e.g., if the
        database is storing data for the state of the system, keys might be
        'positions' and 'velocities'). The values are Numpy arrays with shape
        (number of timestamps, number of samples at each timestamp, [shape of
        variable]).

        Parameters:
            start_datetime (datetime): Beginning of the period we want to fetch
            end_datetime (datetime): End of the period we want to fetch

        Returns:
            (array of float): Numpy array of timestamps encoded as seconds since Unix epoch
            (dict) Time series of associated data
        """
        num_timestamps = len(self._timestamp_list)
        timestamp_array = np.asarray(self._timestamp_list)
        time_series_data = {}
        for variable_name, variable_info in self.structure.items():
            time_series_data[variable_name] = np.empty(
                shape = (num_timestamps, self.num_samples) + tuple(variable_info['shape']),
                dtype = smcmodel.shared_constants._dtypes[variable_info['type']]['numpy']
            )
            for timestamp_index in range(num_timestamps):
                time_series_data[variable_name][timestamp_index] = self._data_dict_list[timestamp_index][variable_name]
        sort_order = np.argsort(timestamp_array)
        timestamp_array = timestamp_array[sort_order]
        for variable_name in self.structure.keys():
            time_series_data[variable_name] = time_series_data[variable_name][sort_order]
        if start_datetime is not None:
            start_timestamp = datetime_conversion.to_posix_timestamp(start_datetime)
            start_datetime_mask = timestamp_array >= start_timestamp
            timestamp_array = timestamp_array[start_datetime_mask]
            for variable_name in self.structure.keys():
                time_series_data[variable_name] = time_series_data[variable_name][start_datetime_mask]
        if end_datetime is not None:
            end_timestamp = datetime_conversion.to_posix_timestamp(end_datetime)
            end_datetime_mask = timestamp_array <= end_timestamp
            timestamp_array = timestamp_array[end_datetime_mask]
            for variable_name in self.structure.keys():
                time_series_data[variable_name] = time_series_data[variable_name][end_datetime_mask]
        return timestamp_array, time_series_data

class DataQueueMemory(DataQueue):
    """
    Implement a DataQueue in memory which supplies data sequentially for a series of timestamps.
    """
    def __init__(self, structure, num_samples, timestamps, time_series_data):
        """
        Constructor for DataQueueMemory class.

        Initializes a dataqueue with time series data.

        The keys of structure should be the variable names associated with the
        object that the database is storing (e.g., if the database is storing
        data for the state of the system, keys might be 'positions' and
        'velocities'). The values should be dicts containing the structure info
        for that those variables: 'shape' and 'type' (see examples of implemented
        SMCModel class for examples of this structure).

        Timestamps should be Python datetime objects (timezone-aware or
        timezone-naive), Numpy datetime64 objects, Pandas Timestamp objects
        (timezone-aware or timezone-naive), strings in ISO format
        (timezone-aware or timezone-naive), or seconds since the Unix epoch
        (float or int).

        Time series data should be in the format returned by Database.fetch_data().

        Parameters:
            structure (dict): Structure of the object that the database is storing
            num_samples (int): Number of samples per timestamp
            timestamps (array of datetime): Timestamps for the data
            time_series_data (dict): Data associated with these timestamps
        """
        timestamps = datetime_conversion.to_posix_timestamps(timestamps)
        num_timestamps = timestamps.shape[0]
        for variable_name, variable_info in structure.items():
            data_shape = time_series_data[variable_name].shape
            if data_shape[0] != num_timestamps:
                raise ValueError('Length of timestamps array is {} but number of time slices in {} is {}'.format(
                    num_timestamps,
                    variable_name,
                    data_shape[0]
                ))
            if data_shape[1] != num_samples:
                raise ValueError('Expected {} samples per timestamp but {} appears to contain {} samples per timestamp'.format(
                    num_resample_indices,
                    variable_name,
                    data_shape[1]
                ))
            if tuple(data_shape[2:]) != tuple(variable_info['shape']):
                raise ValueError('Expected shape for each sample is {} for {} but data appears to be of shape {}'.format(
                    tuple(variable_info['shape']),
                    variable_name,
                    tuple(data_shape[2:])
                ))
        self.timestamps = timestamps
        self.structure = structure
        self.num_samples = num_samples
        self.num_timestamps = num_timestamps
        self.timestamps = timestamps
        self.time_series_data = time_series_data
        self.next_data_pointer = 0

    @classmethod
    def from_database(
        cls,
        database,
        start_datetime = None,
        end_datetime = None
    ):
        """
        Class method for creating a DataQueue from a Database.

        Parameters:
            database (Database): Database containing data for the queue
            start_datetime (datetime): Beginning of the period we want to place in the queue
            end_datetime (datetime): End of the period we want to place in the queue

        Returns:
            (DataQueueMemory): Data queue containing the specified data from the database
        """
        timestamps, time_series_data = database.fetch_data(
            start_datetime,
            end_datetime
        )
        return cls(
            database.structure,
            database.num_samples,
            timestamps,
            time_series_data
        )

    def fetch_next_data(self):
        """
        Fetch data for the next timestamp.

        Data will be fetched in time order.

        The keys of the returned data are the variable names associated with the
        object that the database is holding (e.g., if the database is storing
        data for the state of the system, keys might be 'positions' and
        'velocities'). The values are Numpy arrays with shape (number of samples
        at each timestep, [shape of variable]).

        Once all data in the queue has been fetched, method will return None for
        both outputs.

        Returns:
            (float): Timestamp of the data encoded as seconds since Unix epoch
            (dict): Data associated with that timestamp
        """
        if self.next_data_pointer >= self.num_timestamps:
            return None, None
        else:
            timestamp = self.timestamps[self.next_data_pointer]
            single_time_data  = {}
            for variable_name in self.structure.keys():
                single_time_data[variable_name] = self.time_series_data[variable_name][self.next_data_pointer]
            self.next_data_pointer += 1
            return timestamp, single_time_data
