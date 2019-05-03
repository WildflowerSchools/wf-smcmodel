"""
Define database interfaces for reading and writing data while performing
sequential Monte Carlo (SMC) simulation and inference.
"""

class Database:
    """
    Database for reading and writing data based on timestamp.

    All methods must be implemented by derived classes.
    """
    def write_data(self, timestamp, single_time_data):
        """
        Write data to database with a timestamp.

        The timestamp should be a Python datetime object (timezone-aware or
        timezone-naive), a Numpy datetime64 object, a Pandas Timestamp object
        (timezone-aware or timezone-naive), a string in ISO format
        (timezone-aware or timezone-naive), or seconds since the Unix epoch
        (float or int). If timestamp is timezone-naive, it is assumed to be in
        UTC.

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

    def fetch_data(self, start_datetime = None, end_datetime = None):
        """
        Fetch time series data for a specified time span.

        The datetime arguments should be Python datetime objects (timezone-aware
        or timezone-naive), Numpy datetime64 objects, Pandas Timestamp objects
        (timezone-aware or timezone-naive), strings in ISO format
        (timezone-aware or timezone-naive), or seconds since the Unix epoch
        (float or int). If they are timezone-naive, they are assumed to be in
        UTC.

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
        raise NotImplementedError('Method must be implemented by derived class')

class DataQueue:
    """
    Queue which supplies data sequentially for a series of timestamps.

    All methods must be implemented by derived classes.
    """
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
        raise NotImplementedError('Method must be implemented by derived class')

    def from_database(cls, database, start_datetime = None, end_datetime = None):
        """
        Class method for creating a DataQueue from a Database.

        Parameters:
            database (Database): Database containing data for the queue
            start_datetime (datetime): Beginning of the period we want to place in the queue
            end_datetime (datetime): End of the period we want to place in the queue

        Returns:
            (DataQueue): Data queue containing the specified data from the database
        """
