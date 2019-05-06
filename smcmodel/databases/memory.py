"""
Implement databases in memory for reading and writing time series data while
performing sequential Monte Carlo (SMC) simulation and inference.
"""

import smcmodel.shared_constants
from . import Database
from . import DataQueue
import datetime_conversion
import numpy as np

class DatabaseMemory(Database):
    """
    Implement database in memory for reading and writing time series data.
    """
    def __init__(self, structure, num_samples, timestamps = None, time_series_data = None):
        """
        Constructor for DatabaseMemory class.

        The keys of structure should be the variable names associated with the
        object that the database is storing (e.g., if the database is storing
        data for the state of the system, keys might be 'positions' and
        'velocities'). The values should be dicts containing the structure info
        for that those variables: 'shape' and 'type' (see examples of implemented
        SMCModel class for examples of this structure).

        Timestamps (if specified) should be Python datetime objects (timezone-aware or
        timezone-naive), Numpy datetime64 objects, Pandas Timestamp objects
        (timezone-aware or timezone-naive), strings in ISO format
        (timezone-aware or timezone-naive), or seconds since the Unix epoch
        (float or int).

        The keys of time_series_data (if specified) should match the variable
        names of structure. The values are Numpy arrays with shape (number of
        timestamps, number of samples at each timestamp, [shape of variable]).

        If timestamps and time_series_data are specified, timestamps must be in
        time order and order of data in time_series_data must match timestamps.

        If timestamps and time_series_data are not specified, initializes an
        empty database.

        Parameters:
            structure (dict): Structure of the object that the database is storing
            num_samples (int): Number of samples at each timestamp
            timestamps (array of datetime): Timestamps for the data (default is None)
            time_series_data (dict): Data associated with these timestamps (default is None)
        """
        if timestamps is None and time_series_data is not None:
            raise ValueError('Time series data specified but no timestamps specified')
        if timestamps is not None and time_series_data is None:
            raise ValueError('Timestamps specified but no time series data specified')
        if timestamps is not None and time_series_data is not None:
            timestamps = _convert_timestamps(timestamps)
            num_timestamps = timestamps.shape[0]
            time_series_data = _convert_time_series_data(
                time_series_data,
                structure,
                num_timestamps,
                num_samples
            )
        self.structure = structure
        self.num_samples = num_samples
        self._timestamp_list = []
        self._data_dict_list = []
        if timestamps is not None and time_series_data is not None:
            num_timestamps = timestamps.shape[0]
            for timestamp_index in range(num_timestamps):
                timestamp = timestamps[timestamp_index]
                single_time_data = {}
                for variable_name in structure.keys():
                    single_time_data[variable_name] = time_series_data[variable_name][timestamp_index]
                self.write_data(
                    timestamp,
                    single_time_data)

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
        timestamp = _convert_timestamp(timestamp)
        single_time_data = _convert_single_time_data(single_time_data, self.structure, self.num_samples)
        self._timestamp_list.append(timestamp)
        self._data_dict_list.append(single_time_data)

    def fetch_data(self, start_timestamp = None, end_timestamp = None):
        """
        Fetch time series data for a specified time span.

        The timestamp arguments (if specified) should be Python datetime objects
        (timezone-aware or timezone-naive), Numpy datetime64 objects, Pandas
        Timestamp objects (timezone-aware or timezone-naive), strings in ISO
        format (timezone-aware or timezone-naive), or seconds since the Unix
        epoch (float or int).

        If start_timestamp is not specified, returned data starts with the
        earliest data in the database. If end_timestamp is not specified,
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
            start_timestamp (datetime): Beginning of the period we want to fetch
            end_timestamp (datetime): End of the period we want to fetch

        Returns:
            (array of float): Numpy array of timestamps encoded as seconds since Unix epoch
            (dict) Time series of associated data
        """
        if start_timestamp is not None:
            start_timestamp = _convert_timestamp(start_timestamp)
        if end_timestamp is not None:
            end_timestamp = _convert_timestamp(end_timestamp)
        num_timestamps = len(self._timestamp_list)
        timestamps = np.asarray(self._timestamp_list)
        time_series_data = {}
        for variable_name, variable_info in self.structure.items():
            time_series_data[variable_name] = np.empty(
                shape = (num_timestamps, self.num_samples) + tuple(variable_info['shape']),
                dtype = smcmodel.shared_constants._dtypes[variable_info['type']]['numpy']
            )
            for timestamp_index in range(num_timestamps):
                time_series_data[variable_name][timestamp_index] = self._data_dict_list[timestamp_index][variable_name]
        sort_order = np.argsort(timestamps)
        timestamps = timestamps[sort_order]
        for variable_name in self.structure.keys():
            time_series_data[variable_name] = time_series_data[variable_name][sort_order]
        if start_timestamp is not None:
            start_timestamp_mask = timestamps >= start_timestamp
            timestamps = timestamps[start_timestamp_mask]
            for variable_name in self.structure.keys():
                time_series_data[variable_name] = time_series_data[variable_name][start_timestamp_mask]
        if end_timestamp is not None:
            end_timestamp_mask = timestamps <= end_timestamp
            timestamps = timestamps[end_timestamp_mask]
            for variable_name in self.structure.keys():
                time_series_data[variable_name] = time_series_data[variable_name][end_timestamp_mask]
        return timestamps, time_series_data

    def __iter__(self):
        """
        Create a DataQueueMemory from this database.

        This method will include all data from the database in the data queue.
        If you want to specify a time range, see DataQueue.from_database().

        Returns:
            (DataQueueMemory): Data queue containing the specified data from the database
        """
        timestamps, time_series_data = self.fetch_data()
        return DataQueueMemory(
            self.structure,
            self.num_samples,
            timestamps,
            time_series_data
        )

class DataQueueMemory(DataQueue):
    """
    Implement a DataQueue in memory which supplies data sequentially for a series of timestamps.
    """
    def __init__(self, structure, num_samples, timestamps, time_series_data):
        """
        Constructor for DataQueueMemory class.

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

        The keys of time_series_data should match the variable names of
        structure. The values are Numpy arrays with shape (number of timestamps,
        number of samples at each timestamp, [shape of variable]).

        Timestamps must be in time order and order of data in time_series_data
        must match timestamps.

        Parameters:
            structure (dict): Structure of the object that the database is storing
            num_samples (int): Number of samples per timestamp
            timestamps (array of datetime): Timestamps for the data
            time_series_data (dict): Data associated with these timestamps
        """
        timestamps = _convert_timestamps(timestamps)
        num_timestamps = timestamps.shape[0]
        time_series_data = _convert_time_series_data(
            time_series_data,
            structure,
            num_timestamps,
            num_samples
        )
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
        start_timestamp = None,
        end_timestamp = None
    ):
        """
        Class method for creating a DataQueueMemory from a Database.

        The timestamp arguments (if specified) should be Python datetime objects
        (timezone-aware or timezone-naive), Numpy datetime64 objects, Pandas
        Timestamp objects (timezone-aware or timezone-naive), strings in ISO
        format (timezone-aware or timezone-naive), or seconds since the Unix
        epoch (float or int).

        If start_timestamp is not specified, returned data starts with the
        earliest data in the database. If end_timestamp is not specified,
        returned data ends with the latest data in the database.

        Parameters:
            database (Database): Database containing data for the queue
            start_timestamp (timestamp): Beginning of the period we want to place in the queue
            end_timestamp (timestamp): End of the period we want to place in the queue

        Returns:
            (DataQueueMemory): Data queue containing the specified data from the database
        """
        if start_timestamp is not None:
            start_timestamp = _convert_timestamp(start_timestamp)
        if end_timestamp is not None:
            end_timestamp = _convert_timestamp(end_timestamp)
        timestamps, time_series_data = database.fetch_data(
            start_timestamp,
            end_timestamp
        )
        return cls(
            database.structure,
            database.num_samples,
            timestamps,
            time_series_data
        )

    def __iter__(self):
        return self

    def __next__(self):
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
            raise StopIteration()
        else:
            timestamp = self.timestamps[self.next_data_pointer]
            single_time_data  = {}
            for variable_name in self.structure.keys():
                single_time_data[variable_name] = self.time_series_data[variable_name][self.next_data_pointer]
            self.next_data_pointer += 1
            return timestamp, single_time_data

def _convert_timestamp(timestamp):
        timestamp = datetime_conversion.to_posix_timestamp(timestamp)
        return timestamp

def _convert_timestamps(timestamps):
        timestamps = datetime_conversion.to_posix_timestamps(timestamps)
        return timestamps

def _convert_single_time_data(single_time_data, structure, num_samples):
        if set(single_time_data.keys()) != set(structure.keys()):
            raise ValueError('Variable names in data don\'t match variable names specified in database structure')
        for variable_name, variable_info in structure.items():
            single_time_data[variable_name] = np.asarray(
                single_time_data[variable_name],
                dtype = smcmodel.shared_constants._dtypes[variable_info['type']]['numpy'])
            data_shape = single_time_data[variable_name].shape
            if data_shape[0] != num_samples:
                raise ValueError('Expected {} samples per timestamp but {} appears to contain {} samples per timestamp'.format(
                    num_samples,
                    variable_name,
                    data_shape[0]
                ))
            if tuple(data_shape[1:]) != tuple(variable_info['shape']):
                raise ValueError('Expected shape for each sample is {} for {} but data appears to be of shape {}'.format(
                    tuple(variable_info['shape']),
                    variable_name,
                    tuple(data_shape[1:])
                ))
        return single_time_data

def _convert_time_series_data(time_series_data, structure, num_timestamps, num_samples):
    if set(time_series_data.keys()) != set(structure.keys()):
        raise ValueError('Variable names in data don\'t match variable names specified in database structure')
    for variable_name, variable_info in structure.items():
        time_series_data[variable_name] = np.asarray(
            time_series_data[variable_name],
            dtype = smcmodel.shared_constants._dtypes[variable_info['type']]['numpy'])
        data_shape = time_series_data[variable_name].shape
        if data_shape[0] != num_timestamps:
            raise ValueError('Length of timestamps array is {} but number of time slices in {} is {}'.format(
                num_timestamps,
                variable_name,
                data_shape[0]
            ))
        if data_shape[1] != num_samples:
            raise ValueError('Expected {} samples per timestamp but {} appears to contain {} samples per timestamp'.format(
                num_samples,
                variable_name,
                data_shape[1]
            ))
        if tuple(data_shape[2:]) != tuple(variable_info['shape']):
            raise ValueError('Expected shape for each sample is {} for {} but data appears to be of shape {}'.format(
                tuple(variable_info['shape']),
                variable_name,
                tuple(data_shape[2:])
            ))
    return time_series_data
