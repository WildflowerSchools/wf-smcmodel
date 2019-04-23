# Task list

* Add helper function to convert tensor_dict to tensor_list (for control dependencies)
* Regularize naming of helper functions and their arguments
* Regularize naming between simulate() and estimate_state_trajectory()
* Use iterator for initial value rather than splitting it off
* Eliminate Python variables we don't need (e.g., current_state, etc.)
* Eliminate variables we don't need in simulate() and estimate_state_trajectory()
* Consider adding num_samples argument to simulate()
* Reconsider handline of timestamps (should be datetime objects rather than floats?)
* Consider keeping resampling indices so we can reconstruct trajectories
* Consider adding (optional) summary functions to model definition
* Consider calculating summaries in TensorFlow
* Reconsider whether state trajectory and observation trajectory should be lists of dicts (vs. dicts of expanded arrays)
* Consider creating new classes for state, observation, parameters
* Add type and shape checking
* Add logging
* Add docstrings
* Generate documentation
