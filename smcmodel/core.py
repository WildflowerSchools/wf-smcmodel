import smcmodel.shared_constants
import tensorflow as tf
import numpy as np

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

    def simulate_trajectory(self, datetimes, state_database, observation_database):
        # Convert datetimes to Numpy array of (micro)seconds since epoch
        timestamps = _datetimes_to_timestamps_array(datetimes)
        # Build the dataflow graph
        simulation_graph = tf.Graph()
        with simulation_graph.as_default():
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
            time = tf.placeholder(dtype = tf.float64, shape = [], name = 'time')
            next_time = tf.placeholder(dtype = tf.float64, shape = [], name = 'next_time')
            next_state = self.transition_model_sample(
                state,
                time,
                next_time,
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
        with tf.Session(graph=simulation_graph) as sess:
            # Initialize the persistent variables
            sess.run(init)
            # Calculate and store the initial state and initial observation
            initial_state_value, initial_observation_value = sess.run([state, initial_observation])
            state_database.write_data(timestamps[0], initial_state_value)
            observation_database.write_data(timestamps[0], initial_observation_value)
            # Calculate and store the state and observation for all subsequent time steps
            for timestamp_index in range(1, num_timestamps):
                time_value = timestamps[timestamp_index - 1]
                next_time_value = timestamps[timestamp_index]
                next_state_value, next_observation_value = sess.run(
                    [assign_state, next_observation],
                    feed_dict = {time: time_value, next_time: next_time_value}
                )
                state_database.write_data(timestamps[timestamp_index], next_state_value)
                observation_database.write_data(timestamps[timestamp_index], next_observation_value)

    def estimate_state_trajectory(
        self,
        num_particles,
        observation_data_queue,
        state_summary_database):
        # Build the dataflow graph
        state_trajectory_estimation_graph = tf.Graph()
        with state_trajectory_estimation_graph.as_default():
            # Sample the global parameters
            parameters = self.parameter_model_sample()
            # Calculate the initial values for the persistent variables
            initial_observation = _placeholder_dict(self.observation_structure)
            initial_state = self.initial_model_sample(
                num_particles,
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
            time = tf.placeholder(dtype = tf.float64, shape = [], name = 'time')
            next_time = tf.placeholder(dtype = tf.float64, shape = [], name = 'next_time')
            next_observation = _placeholder_dict(self.observation_structure)
            resample_indices = tf.squeeze(
                tf.random.categorical(
                    [log_weights],
                    num_particles
                )
            )
            state_resampled = _resample_tensor_dict(
                self.state_structure,
                state,
                resample_indices)
            next_state = self.transition_model_sample(
                state_resampled,
                time,
                next_time,
                parameters
            )
            next_log_weights = self.observation_model_pdf(
                next_state,
                next_observation,
                parameters)
            next_state_summary = self.state_summary(
                next_state,
                next_log_weights,
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
        with tf.Session(graph=state_trajectory_estimation_graph) as sess:
            # Calculate initial values and initialize the persistent variables
            initial_time_value, initial_observation_value = observation_data_queue.fetch_next_data()
            initial_observation_feed_dict = _feed_dict(
                self.observation_structure,
                initial_observation,
                initial_observation_value
            )
            initial_state_summary_value, _ = sess.run(
                [initial_state_summary, init],
                feed_dict = initial_observation_feed_dict)
            state_summary_database.write_data(initial_time_value, initial_state_summary_value)
            # Calculate and store the state samples and log weights for all subsequent time steps
            time_value = initial_time_value
            while True:
                next_time_value, next_observation_value = observation_data_queue.fetch_next_data()
                if next_time_value is None:
                    break
                time_feed_dict = {time: time_value, next_time: next_time_value}
                next_operation_feed_dict = _feed_dict(
                    self.observation_structure,
                    next_observation,
                    next_observation_value
                )
                combined_feed_dict = {**time_feed_dict, **next_operation_feed_dict}
                next_state_summary_value, _, _ = sess.run(
                    [next_state_summary, assign_state, assign_log_weights],
                    feed_dict = combined_feed_dict
                )
                state_summary_database.write_data(next_time_value, next_state_summary_value)
                time_value = next_time_value

def _placeholder_dict(structure):
    placeholder_dict = {}
    for variable_name, variable_info in structure.items():
        placeholder_dict[variable_name] = tf.placeholder(
            dtype = smcmodel.shared_constants._dtypes[variable_info['type']]['tensorflow'],
            shape = tuple([1]) + tuple(variable_info['shape'])
        )
    return placeholder_dict

def _feed_dict(structure, tensor_dict, array_dict):
    feed_dict = {}
    for variable_name in structure.keys():
        feed_dict[tensor_dict[variable_name]] = array_dict[variable_name]
    return feed_dict

def _to_array_dict(structure, input):
    array_dict = {}
    for variable_name, variable_info in structure.items():
        array_dict[variable_name] = np.asarray(
            input[variable_name],
            dtype = smcmodel.shared_constants._dtypes[variable_info['type']]['numpy']
        )
    return array_dict

def _array_dict_to_tensor_dict(structure, array_dict):
    tensor_dict = {}
    for variable_name, variable_info in structure.items():
        tensor_dict[variable_name] = tf.constant(
            array_dict[variable_name],
            dtype = smcmodel.shared_constants._dtypes[variable_info['type']]['tensorflow']
        )
    return array_dict

def _datetimes_to_timestamps_array(datetimes):
    timestamps_array = np.asarray(
        [datetime.timestamp() for datetime in datetimes],
        dtype = np.float64
    )
    return timestamps_array

def _get_variable_dict(structure, initial_values):
    variable_dict = {}
    for variable_name, variable_info in structure.items():
        variable_dict[variable_name] = tf.get_variable(
            name = variable_name,
            dtype = smcmodel.shared_constants._dtypes[variable_info['type']]['tensorflow'],
            initializer = initial_values[variable_name])
    return variable_dict

def _variable_dict_assign(structure, variable_dict, values):
    assign_dict = {}
    for variable_name in structure.keys():
        assign_dict[variable_name] = variable_dict[variable_name].assign(values[variable_name])
    return assign_dict

def _array_dict_to_iterator_dict(structure, array_dict):
    tensor_dict = _array_dict_to_tensor_dict(structure, array_dict)
    iterator_dict={}
    for variable_name in structure.keys():
        dataset = tf.data.Dataset.from_tensor_slices(tensor_dict[variable_name])
        iterator_dict[variable_name] = dataset.make_one_shot_iterator()
    return iterator_dict

def _iterator_dict_get_next(structure, iterator_dict):
    tensor_dict = {}
    for variable_name in structure.keys():
        tensor_dict[variable_name] = iterator_dict[variable_name].get_next()
    return tensor_dict

def _resample_tensor_dict(structure, tensor_dict, resample_indices):
    tensor_dict_resampled = {}
    for variable_name in structure.keys():
        tensor_dict_resampled[variable_name] = tf.gather(
            tensor_dict[variable_name],
            resample_indices
        )
    return tensor_dict_resampled

def _tensor_list(tensor_dict):
    return list(tensor_dict.values())

def _initialize_trajectory(num_timestamps, num_samples, structure, initial_values):
    trajectory = {
        variable_name: np.zeros(
            (num_timestamps, num_samples) + tuple(variable_info['shape'])
        )
        for variable_name, variable_info
        in structure.items()
    }
    for variable_name in structure.keys():
        trajectory[variable_name][0] = initial_values[variable_name]
    return trajectory

def _extend_trajectory(trajectory, timestamp_index, structure, values):
    for variable_name in structure.keys():
        trajectory[variable_name][timestamp_index] = values[variable_name]
    return trajectory
