import smcmodel

class Database:

    def write_data(self, datetime, data_dict):
        raise NotImplementedError('Method must be implemented by derived class')

    def fetch_data(self, start_datetime = None, end_datetime = None):
        raise NotImplementedError('Method must be implemented by derived class')

class DataQueue:
    def __init__(self, structure):
        raise NotImplementedError('Method must be implemented by derived class')
    def read_next_data():
        raise NotImplementedError('Method must be implemented by derived class')
