"""
Define database interfaces for reading and writing data while performing
sequential Monte Carlo (SMC) simulation and inference.
"""

class Database:
    """
    Implement database for reading and writing time series data.
    """
    def __init__(self, structure, num_samples, timestamps = None, time_series_data = None):
        """
        Constructor for Database class.

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

        If timestamps and time_series_data are not specified, initializes an
        empty database.

        Parameters:
            structure (dict): Structure of the object that the database is storing
            num_samples (int): Number of samples at each timestamp
            timestamps (array of datetime): Timestamps for the data (default is None)
            time_series_data (dict): Data associated with these timestamps (default is None)
        """
        raise NotImplementedError('Method must be implemented by derived class')

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
        raise NotImplementedError('Method must be implemented by derived class')

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
        raise NotImplementedError('Method must be implemented by derived class')

    def __iter__(self):
        """
        Create a DataQueue from this database.

        This method will include all data from the database in the data queue.
        If you want to specify a time range, see DataQueue.from_database().

        Returns:
            (DataQueue): Data queue containing the specified data from the database
        """
        raise NotImplementedError('Method must be implemented by derived class')

class DataQueue:
    """
    Implement a data queue which supplies data sequentially for a series of timestamps.
    """
    def __init__(self, structure, num_samples, timestamps, time_series_data):
        """
        Constructor for DataQueue class.

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

        Parameters:
            structure (dict): Structure of the object that the database is storing
            num_samples (int): Number of samples per timestamp
            timestamps (array of datetime): Timestamps for the data
            time_series_data (dict): Data associated with these timestamps
        """
        raise NotImplementedError('Method must be implemented by derived class')

    @classmethod
    def from_database(
        cls,
        database,
        start_timestamp = None,
        end_timestamp = None
    ):
        """
        Class method for creating a DataQueue from a Database.

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
            (DataQueue): Data queue containing the specified data from the database
        """
        raise NotImplementedError('Method must be implemented by derived class')

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
        raise NotImplementedError('Method must be implemented by derived class')
