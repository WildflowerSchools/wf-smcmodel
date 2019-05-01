# Task list

* Figure out how to force assignment ops without fetching their outputs
* Add database functionality
* Add explicit names to tensors and ops
* Check dataflow graphs using tensorboard (are all objects of known size at graph specification time)
* Add type and shape checking
* Add logging
* Add docstrings
* Generate documentation
* Make functions handle datetimes in an array of shapes/types/formats
* Make functions handle observation trajectories in an array of shapes/types/formats
* Consider adding (optional) summary functions to model definition
* Consider calculating summaries in TensorFlow
* Consider adding num_samples argument to simulate()
* Reconsider whether state trajectory and observation trajectory should be lists of dicts (vs. dicts of expanded arrays)
* Consider creating new classes for state, observation, parameters
