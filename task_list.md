# Task list

* Add num_samples to placeholder_dict()
* Check keys in all functions that build/convert objects
* Consider moving helper functions to top level
* In simulate, do init and pull initial values in one run
* Call them samples rather than particles
* Add logging
* Add docstrings
* Generate documentation
* Add explicit names to tensors and ops
* Check dataflow graphs using tensorboard (are all objects of known size at graph specification time)
* Figure out how to force assignment ops without fetching their outputs
* Consider adding num_samples argument to simulate()
