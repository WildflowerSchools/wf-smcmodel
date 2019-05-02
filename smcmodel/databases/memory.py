import smcmodel.shared_constants
from . import Database
import datetime_conversion
import numpy as np

class DatabaseMemory(Database):

    def __init__(self, structure, num_samples):
        self.structure = structure
        self.num_samples = num_samples
        self._timestamp_list = []
        self._data_dict_list = []

    def write_data(self, datetime, data_dict):
        timestamp = datetime_conversion.to_posix_timestamp(datetime)
        if set(data_dict.keys()) != set(self.structure.keys()):
            raise ValueError('Keys in new data don\'t match variable structure of database')
        for variable_name in self.structure.keys():
            data_dict[variable_name] = np.asarray(data_dict[variable_name])
            array_num_samples = data_dict[variable_name].shape[0]
            if array_num_samples != self.num_samples:
                raise ValueError('Database expects {} samples but new data for {} appears to contain {} samples'.format(
                    self.num_samples,
                    variable_name,
                    array_num_samples
                ))
        self._timestamp_list.append(timestamp)
        self._data_dict_list.append(data_dict)

    def fetch_data(self, start_datetime = None, end_datetime = None):
        num_timestamps = len(self._timestamp_list)
        timestamp_array = np.asarray(self._timestamp_list)
        trajectory_dict = {}
        for variable_name, variable_info in self.structure.items():
            trajectory_dict[variable_name] = np.empty(
                shape = (num_timestamps, self.num_samples) + tuple(variable_info['shape']),
                dtype = smcmodel.shared_constants._dtypes[variable_info['type']]['numpy']
            )
            for timestamp_index in range(num_timestamps):
                trajectory_dict[variable_name][timestamp_index] = self._data_dict_list[timestamp_index][variable_name]
        sort_order = np.argsort(timestamp_array)
        timestamp_array = timestamp_array[sort_order]
        for variable_name in self.structure.keys():
            trajectory_dict[variable_name] = trajectory_dict[variable_name][sort_order]
        if start_datetime is not None:
            start_timestamp = datetime_conversion.to_posix_timestamp(start_datetime)
            start_datetime_mask = timestamp_array >= start_timestamp
            timestamp_array = timestamp_array[start_datetime_mask]
            for variable_name in self.structure.keys():
                trajectory_dict[variable_name] = trajectory_dict[variable_name][start_datetime_mask]
        if end_datetime is not None:
            end_timestamp = datetime_conversion.to_posix_timestamp(end_datetime)
            end_datetime_mask = timestamp_array <= end_timestamp
            timestamp_array = timestamp_array[end_datetime_mask]
            for variable_name in self.structure.keys():
                trajectory_dict[variable_name] = trajectory_dict[variable_name][end_datetime_mask]
        return timestamp_array, trajectory_dict

class DataQueue:
    def __init__(self, structure):
        raise NotImplementedError('Method must be implemented by derived class')
    def read_next_data():
        raise NotImplementedError('Method must be implemented by derived class')
