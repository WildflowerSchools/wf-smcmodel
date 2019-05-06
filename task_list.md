# Task list

* Maybe use consistent language for single sample, single time slice, multiple time slices
* Use new datetime conversion functions instead of datetimes_to_timestamps_array()
* Call them timestamps rather than times or datetimes
* Add num_samples to placeholder_dict()
* Consistently use to or not in conversion functions
* Check keys in all functions that build/convert objects
* Consider moving helper functions to top level
* In array_dict_to_tensor_dict, convert to array first (and rename)
* Get rid of trajectory functions
* Call it single_time_data and time_series_data everywhere (instead of data and trajectory)
* In simulate, do init and pull initial values in one run
* Call them samples rather than particles
* Add logging
* Add docstrings
* Generate documentation
* Add explicit names to tensors and ops
* Check dataflow graphs using tensorboard (are all objects of known size at graph specification time)
* Figure out how to force assignment ops without fetching their outputs
* Consider adding num_samples argument to simulate()
