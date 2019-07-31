"""
Define pipes for reading and writing data while performing sequential Monte
Carlo (SMC) simulation and inference.
"""
import smcmodel.shared_constants
import datetime
import dateutil.parser
import numpy as np


class DataSource:
    """
    Pipe for pulling data into the inference engine.
    """

    def __iter__(self):
        return self

    def __next__(self):
        """
        Fetch data for the next timestamp.

        Data will be fetched in time order.

        The keys of the returned data are the variable names associated with the
        type of data that is being pulled in (e.g., if we are pulling in
        observation data, the keys might be 'rssi' and 'acceleration'). The
        values are Numpy arrays with shape (number of samples at each timestep,
        [shape of variable]).

        Once all data in the queue has been fetched, method will raise a
        StopIteration exception.

        Returns:
            (float): Timestamp of the data encoded as seconds since Unix epoch
            (dict): Data associated with that timestamp
        """
        return self._next()

    def _next(self):
        raise NotImplementedError('Method must be implemented by derived class')

class DataDestination:
    """
    Pipe for pushing data out of the inference engine.
    """

    def write_data(self, timestamp, single_time_data):
        """
        Write data with a timestamp.

        Timestamp should in seconds since epoch.

        The keys of single_time_data should be the variable names associated
        with the type of data that is being pushed out (e.g., if the database is
        storing data for the state of the system, keys might be 'positions' and
        'velocities'). The values should be array-like (convertible to Numpy
        arrays) with shape (number of samples at each timestep, [shape of
        variable]).

        Parameters:
            timestamp (float): Timestamp associated with the data
            single_time_data (dict): Data for that timestamp
        """
        return self._write_data(timestamp, single_time_data)

    def _write_data(self, timestamp, single_time_data):
        raise NotImplementedError('Method must be implemented by derived class')

class DataSourceArrayDict(DataSource):
    """
    Pipe for pulling data into the inference engine from a dict of arrays.
    """

    def __init__(self, structure, num_samples, timestamps, array_dict):
        try:
            timestamps_parsed = np.asarray(timestamps, dtype = np.float)
        except:
            raise ValueError('Timestamps could not be parsed as an array of floats')
        if timestamps_parsed.ndim != 1:
            raise ValueError('Timestamps must be a one-dimensional array')
        num_timestamps = timestamps_parsed.shape[0]
        array_dict_parsed = {}
        for variable_name in structure.keys():
            variable_type = structure[variable_name]['type']
            variable_shape = structure[variable_name]['shape']
            variable_dtype_numpy = smcmodel.shared_constants._dtypes[variable_type]['numpy']
            if variable_name not in array_dict.keys():
                raise ValueError('Variable {} specified in structure but not found in array dict')
            variable_value = array_dict[variable_name]
            try:
                array_dict_parsed[variable_name] = np.asarray(
                    variable_value,
                    dtype = variable_dtype_numpy
                )
            except:
                raise ValueError('Variable {} cannot be parsed as array of dtype {}'.format(
                    variable_name,
                    variable_dtype_numpy
                ))
            expected_array_shape = (num_timestamps, num_samples) + tuple(variable_shape)
            array_shape = array_dict_parsed[variable_name].shape
            if array_shape != expected_array_shape:
                raise ValueError('Expected shape {} for {} but received shape {}'.format(
                    expected_array_shape,
                    variable_name,
                    array_shape
                ))
            self.structure = structure
            self.num_samples = num_samples
            self.num_timestamps = num_timestamps
            self.timestamps = timestamps_parsed
            self.array_dict = array_dict_parsed
            self.timestamp_index = 0

    def _next(self):
        if self.timestamp_index >= self.num_timestamps:
            raise StopIteration()
        else:
            timestamp = self.timestamps[self.timestamp_index]
            single_time_data  = {}
            for variable_name in self.structure.keys():
                single_time_data[variable_name] = self.array_dict[variable_name][self.timestamp_index]
            self.timestamp_index += 1
            return timestamp, single_time_data

class DataDestinationArrayDict(DataDestination):
    """
    Pipe for pushing data out of the inference engine into a dict of arrays.
    """

    def __init__(self, structure, num_samples):
        self.timestamps = np.empty(shape = (0,), dtype = np.float)
        self.array_dict = {}
        for variable_name in structure.keys():
            variable_type = structure[variable_name]['type']
            variable_shape = structure[variable_name]['shape']
            variable_dtype_numpy = smcmodel.shared_constants._dtypes[variable_type]['numpy']
            array_shape = (0, num_samples) + tuple(variable_shape)
            self.array_dict[variable_name] = np.empty(
                shape = array_shape,
                dtype = variable_dtype_numpy
            )
        self.structure = structure
        self.num_samples = num_samples

    def _write_data(self, timestamp, single_time_data):
        try:
            timestamp_parsed = np.asarray(timestamp, dtype = np.float)
        except:
            raise ValueError('Timestamp could not be parsed as a float')
        if timestamp_parsed.size != 1:
            raise ValueError('Expected a single timestamp but received {}'.format(timestamp_parsed.size))
        self.timestamps = np.concatenate((
            self.timestamps,
            np.expand_dims(timestamp_parsed, axis = 0)
        ))
        for variable_name in self.structure.keys():
            variable_type = self.structure[variable_name]['type']
            variable_shape = self.structure[variable_name]['shape']
            variable_dtype_numpy = smcmodel.shared_constants._dtypes[variable_type]['numpy']
            if variable_name not in single_time_data.keys():
                raise ValueError('Variable {} specified in structure but not found in data')
            variable_value = single_time_data[variable_name]
            try:
                variable_value_parsed = np.asarray(
                    variable_value,
                    dtype = variable_dtype_numpy
                )
            except:
                raise ValueError('Variable {} cannot be parsed as array of dtype {}'.format(
                    variable_name,
                    variable_dtype_numpy
                ))
            expected_array_shape = (self.num_samples,) + tuple(variable_shape)
            array_shape = variable_value_parsed.shape
            if array_shape != expected_array_shape:
                raise ValueError('Expected shape {} for {} but received shape {}'.format(
                    expected_array_shape,
                    variable_name,
                    array_shape
                ))
            self.array_dict[variable_name] = np.concatenate((
                self.array_dict[variable_name],
                np.expand_dims(variable_value_parsed, axis = 0)
            ))
