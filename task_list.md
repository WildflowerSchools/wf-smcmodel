# Task list

* Reconsider handline of timestamps (should be datetime objects rather than floats?)
* Consistently rename previous time and time to time and next time
* Consider adding num_samples argument to simulate()
* Consider keeping resampling indices so we can reconstruct trajectories
* Consider adding (optional) summary functions to model definition
* Consider calculating summaries in TensorFlow
* Reconsider whether state trajectory and observation trajectory should be lists of dicts (vs. dicts of expanded arrays)
* Consider creating new classes for state, observation, parameters
* Add explicit names to tensors and ops
* Check dataflow graphs using tensorboard (are all objects of known size at graph specification time)
* Add type and shape checking
* Add logging
* Add docstrings
* Generate documentation
