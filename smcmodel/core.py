import smcmodel.shared_constants
import tensorflow as tf
import numpy as np
import tqdm
import datetime_conversion

class SMCModelGeneralTensorflow:

    def __init__(
        self,
        parameter_structure = None,
        state_structure = None,
        observation_structure = None,
        state_summary_structure = None,
        parameter_model_sample = None,
        initial_model_sample = None,
        transition_model_sample = None,
        observation_model_sample = None,
        observation_model_pdf = None,
        state_summary = None):
        self.parameter_structure = parameter_structure
        self.state_structure = state_structure
        self.observation_structure = observation_structure
        self.state_summary_structure = state_summary_structure
        self.parameter_model_sample = parameter_model_sample
        self.initial_model_sample = initial_model_sample
        self.transition_model_sample = transition_model_sample
        self.observation_model_sample = observation_model_sample
        self.observation_model_pdf = observation_model_pdf
        self.state_summary = state_summary

    def simulate_time_series(self, timestamps, state_database, observation_database):
        # Convert timestamps to Numpy array of (micro)seconds since epoch
        timestamps = datetime_conversion.to_posix_timestamps(timestamps)
        # Build the dataflow graph
        simulate_time_series_graph = tf.Graph()
        with simulate_time_series_graph.as_default():
            # Sample the global parameters
            parameters = self.parameter_model_sample()
            # Calculate the initial values for the persistent variables
            initial_state = self.initial_model_sample(1, parameters)
            initial_observation = self.observation_model_sample(initial_state, parameters)
            # Define the persistent variables
            state = _get_variable_dict(self.state_structure, initial_state)
            # Initialize the persistent variables
            init = tf.global_variables_initializer()
            # Calculate the next time step in the simulation
            timestamp = tf.placeholder(dtype = tf.float64, shape = [], name = 'timestamp')
            next_timestamp = tf.placeholder(dtype = tf.float64, shape = [], name = 'next_timestamp')
            next_state = self.transition_model_sample(
                state,
                timestamp,
                next_timestamp,
                parameters)
            next_observation = self.observation_model_sample(next_state, parameters)
            # Assign these values to the persistent variables so they become the inputs for the next time step
            control_dependencies = _tensor_list(next_state)
            with tf.control_dependencies(control_dependencies):
                assign_state = _variable_dict_assign(
                    self.state_structure,
                    state,
                    next_state
                )
        # Run the calcuations using the graph above
        num_timestamps = timestamps.shape[0]
        with tf.Session(graph=simulate_time_series_graph) as sess:
            # Initialize the persistent variables
            sess.run(init)
            # Calculate and store the initial state and initial observation
            initial_state_value, initial_observation_value = sess.run([state, initial_observation])
            state_database.write_data(timestamps[0], initial_state_value)
            observation_database.write_data(timestamps[0], initial_observation_value)
            # Calculate and store the state and observation for all subsequent time steps
            for timestamp_index in range(1, num_timestamps):
                timestamp_value = timestamps[timestamp_index - 1]
                next_timestamp_value = timestamps[timestamp_index]
                next_state_value, next_observation_value = sess.run(
                    [assign_state, next_observation],
                    feed_dict = {timestamp: timestamp_value, next_timestamp: next_timestamp_value}
                )
                state_database.write_data(timestamps[timestamp_index], next_state_value)
                observation_database.write_data(timestamps[timestamp_index], next_observation_value)

    def estimate_state_time_series(
        self,
        num_samples,
        observation_data_queue,
        state_summary_database,
        progress_bar = False,
        num_observations = None):
        # Build the dataflow graph
        state_time_series_estimation_graph = tf.Graph()
        with state_time_series_estimation_graph.as_default():
            # Sample the global parameters
            parameters = self.parameter_model_sample()
            # Calculate the initial values for the persistent variables
            initial_observation = _placeholder_dict(self.observation_structure)
            initial_state = self.initial_model_sample(
                num_samples,
                parameters
            )
            initial_log_weights = self.observation_model_pdf(
                initial_state,
                initial_observation,
                parameters
            )
            initial_state_summary = self.state_summary(
                initial_state,
                initial_log_weights,
                tf.zeros(shape = [num_samples], dtype = tf.int64),
                parameters
            )
            # Define the persistent variables
            state = _get_variable_dict(self.state_structure, initial_state)
            log_weights = tf.get_variable(
                name='log_weights',
                dtype = tf.float32,
                initializer = initial_log_weights
            )
            # Initialize the persistent variables
            init = tf.global_variables_initializer()
            # Calculate the state samples and log weights for the next time step
            timestamp = tf.placeholder(dtype = tf.float64, shape = [], name = 'timestamp')
            next_timestamp = tf.placeholder(dtype = tf.float64, shape = [], name = 'next_timestamp')
            next_observation = _placeholder_dict(self.observation_structure)
            resample_indices = tf.squeeze(
                tf.random.categorical(
                    [log_weights],
                    num_samples
                )
            )
            state_resampled = _resample_tensor_dict(
                self.state_structure,
                state,
                resample_indices)
            next_state = self.transition_model_sample(
                state_resampled,
                timestamp,
                next_timestamp,
                parameters
            )
            next_log_weights = self.observation_model_pdf(
                next_state,
                next_observation,
                parameters)
            next_state_summary = self.state_summary(
                next_state,
                next_log_weights,
                resample_indices,
                parameters
            )
            # Assign these values to the persistent variables so they become the inputs for the next time step
            control_dependencies = _tensor_list(next_state) + _tensor_list(next_state_summary) + [next_log_weights]
            with tf.control_dependencies(control_dependencies):
                assign_state = _variable_dict_assign(
                    self.state_structure,
                    state,
                    next_state
                )
                assign_log_weights = log_weights.assign(next_log_weights)
        # Run the calcuations using the graph above
        with tf.Session(graph=state_time_series_estimation_graph) as sess:
            # Calculate initial values and initialize the persistent variables
            initial_timestamp_value, initial_observation_value = next(observation_data_queue)
            initial_observation_feed_dict = _feed_dict(
                self.observation_structure,
                initial_observation,
                initial_observation_value
            )
            initial_state_summary_value, _ = sess.run(
                [initial_state_summary, init],
                feed_dict = initial_observation_feed_dict)
            state_summary_database.write_data(initial_timestamp_value, initial_state_summary_value)
            # Calculate and store the state samples and log weights for all subsequent time steps
            timestamp_value = initial_timestamp_value
            # Set up the progress bar (if set)
            if progress_bar:
                if num_observations is not None:
                    remaining_iterations = num_observations - 1
                else:
                    remaining_iterations = None
                t = tqdm.tqdm(total = remaining_iterations)
            for next_timestamp_value, next_observation_value in observation_data_queue:
                timestamp_feed_dict = {timestamp: timestamp_value, next_timestamp: next_timestamp_value}
                next_operation_feed_dict = _feed_dict(
                    self.observation_structure,
                    next_observation,
                    next_observation_value
                )
                combined_feed_dict = {**timestamp_feed_dict, **next_operation_feed_dict}
                next_state_summary_value, _, _ = sess.run(
                    [next_state_summary, assign_state, assign_log_weights],
                    feed_dict = combined_feed_dict
                )
                state_summary_database.write_data(next_timestamp_value, next_state_summary_value)
                timestamp_value = next_timestamp_value
                # Update the progress bar (if set)
                if progress_bar:
                    t.update()
            # Close the progress bar (if set)
            if progress_bar:
                t.close()

def _placeholder_dict(structure, num_samples = 1):
    placeholder_dict = {}
    for variable_name, variable_info in structure.items():
        placeholder_dict[variable_name] = tf.placeholder(
            dtype = smcmodel.shared_constants._dtypes[variable_info['type']]['tensorflow'],
            shape = tuple([num_samples]) + tuple(variable_info['shape'])
        )
    return placeholder_dict

def _to_array_dict(structure, input):
    if set(input.keys()) != set(structure.keys()):
        raise ValueError('Variable names in data don\'t match variable names specified in database structure')
    array_dict = {}
    for variable_name, variable_info in structure.items():
        array_dict[variable_name] = np.asarray(
            input[variable_name],
            dtype = smcmodel.shared_constants._dtypes[variable_info['type']]['numpy']
        )
    return array_dict

def _to_tensor_dict(structure, input):
    if set(input.keys()) != set(structure.keys()):
        raise ValueError('Variable names in data don\'t match variable names specified in database structure')
    array_dict = _to_array_dict(structure, input)
    tensor_dict = {}
    for variable_name, variable_info in structure.items():
        tensor_dict[variable_name] = tf.constant(
            array_dict[variable_name],
            dtype = smcmodel.shared_constants._dtypes[variable_info['type']]['tensorflow']
        )
    return array_dict

def _to_iterator_dict(structure, input):
    if set(input.keys()) != set(structure.keys()):
        raise ValueError('Variable names in data don\'t match variable names specified in database structure')
    tensor_dict = _to_tensor_dict(structure, input)
    iterator_dict={}
    for variable_name in structure.keys():
        dataset = tf.data.Dataset.from_tensor_slices(tensor_dict[variable_name])
        iterator_dict[variable_name] = dataset.make_one_shot_iterator()
    return iterator_dict

def _feed_dict(structure, tensor_dict, input):
    if set(input.keys()) != set(structure.keys()):
        raise ValueError('Variable names in data don\'t match variable names specified in database structure')
    array_dict = _to_array_dict(structure, input)
    feed_dict = {}
    for variable_name in structure.keys():
        feed_dict[tensor_dict[variable_name]] = array_dict[variable_name]
    return feed_dict

def _get_variable_dict(structure, initial_values_tensor_dict):
    if set(initial_values_tensor_dict.keys()) != set(structure.keys()):
        raise ValueError('Variable names in data don\'t match variable names specified in database structure')
    variable_dict = {}
    for variable_name, variable_info in structure.items():
        variable_dict[variable_name] = tf.get_variable(
            name = variable_name,
            dtype = smcmodel.shared_constants._dtypes[variable_info['type']]['tensorflow'],
            initializer = initial_values_tensor_dict[variable_name])
    return variable_dict

def _variable_dict_assign(structure, variable_dict, values_tensor_dict):
    if set(variable_dict.keys()) != set(structure.keys()):
        raise ValueError('Variable names in variable dict don\'t match variable names specified in database structure')
    if set(values_tensor_dict.keys()) != set(structure.keys()):
        raise ValueError('Variable names in data don\'t match variable names specified in database structure')
    assign_dict = {}
    for variable_name in structure.keys():
        assign_dict[variable_name] = variable_dict[variable_name].assign(values_tensor_dict[variable_name])
    return assign_dict

def _iterator_dict_get_next(structure, iterator_dict):
    if set(iterator_dict.keys()) != set(structure.keys()):
        raise ValueError('Variable names in iterator_dict don\'t match variable names specified in database structure')
    tensor_dict = {}
    for variable_name in structure.keys():
        tensor_dict[variable_name] = iterator_dict[variable_name].get_next()
    return tensor_dict

def _resample_tensor_dict(structure, tensor_dict, resample_indices):
    if set(tensor_dict.keys()) != set(structure.keys()):
        raise ValueError('Variable names in tensor dict don\'t match variable names specified in database structure')
    tensor_dict_resampled = {}
    for variable_name in structure.keys():
        tensor_dict_resampled[variable_name] = tf.gather(
            tensor_dict[variable_name],
            resample_indices
        )
    return tensor_dict_resampled

def _tensor_list(tensor_dict):
    return list(tensor_dict.values())

def _initialize_time_series(num_timestamps, num_samples, structure, initial_values):
    time_series = {
        variable_name: np.zeros(
            (num_timestamps, num_samples) + tuple(variable_info['shape'])
        )
        for variable_name, variable_info
        in structure.items()
    }
    for variable_name in structure.keys():
        time_series[variable_name][0] = initial_values[variable_name]
    return time_series

def _extend_time_series(time_series, timestamp_index, structure, values):
    for variable_name in structure.keys():
        time_series[variable_name][timestamp_index] = values[variable_name]
    return time_series
