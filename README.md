# smcmodel

Provide estimation and simulation capabilities for sequential Monte Carlo (SMC) models

## Task list

* Add logging
* Add docstrings
* Generate documentation
* Add explicit names to tensors and ops
* Check dataflow graphs using tensorboard (are all objects of known size at graph specification time)
* In simulate, do init and pull initial values in one run
* Figure out how to force assignment ops without fetching their outputs
* Consider adding num_samples argument to simulate()
