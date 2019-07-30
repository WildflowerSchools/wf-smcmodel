"""
Define pipes for reading and writing data while performing sequential Monte
Carlo (SMC) simulation and inference.
"""
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
