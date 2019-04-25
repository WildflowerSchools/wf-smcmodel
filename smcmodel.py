import tensorflow as tf
import numpy as np

class SMCModelGeneralTensorflow:

    def __init__(
        self,
        parameter_structure = None,
        state_structure = None,
        observation_structure = None,
        parameter_model_sample = None,
        initial_model_sample = None,
        transition_model_sample = None,
        observation_model_sample = None,
        observation_model_pdf = None):
        self.parameter_structure = parameter_structure
        self.state_structure = state_structure
        self.observation_structure = observation_structure
        self.parameter_model_sample = parameter_model_sample
        self.initial_model_sample = initial_model_sample
        self.transition_model_sample = transition_model_sample
        self.observation_model_sample = observation_model_sample
        self.observation_model_pdf = observation_model_pdf

    def simulate_trajectory(self, timestamps):
        # Build the dataflow graph
        simulation_graph = tf.Graph()
        with simulation_graph.as_default():
            # Sample the global parameters
            parameters = self.parameter_model_sample()
            # Calculate the initial values for the persistent variables
            timestamps_iterator = tf.data.Dataset.from_tensor_slices(timestamps).make_one_shot_iterator()
            initial_time = timestamps_iterator.get_next()
            initial_state = self.initial_model_sample(1, parameters)
            initial_observation = self.observation_model_sample(initial_state, parameters)
            # Define the persistent variables
            time = tf.get_variable(
                name = 'time',
                dtype = tf.float32,
                initializer = initial_time)
            state = _get_variable_dict(self.state_structure, initial_state)
            observation = _get_variable_dict(self.observation_structure, initial_observation)
            # Initialize the persistent variables
            init = tf.global_variables_initializer()
            # Calculate the next time step in the simulation
            next_time = timestamps_iterator.get_next()
            next_state = self.transition_model_sample(
                state,
                time,
                next_time,
                parameters)
            next_observation = self.observation_model_sample(next_state, parameters)
            # Assign these values to the persistent variables so they become the inputs for the next time step
            control_dependencies = [next_time] + _tensor_list(next_state) + _tensor_list(next_observation)
            with tf.control_dependencies(control_dependencies):
                assign_time = time.assign(next_time)
                assign_state = _variable_dict_assign(
                    self.state_structure,
                    state,
                    next_state
                )
                assign_observation = _variable_dict_assign(
                    self.observation_structure,
                    observation,
                    next_observation
                )
        # Run the calcuations using the graph above
        num_timestamps = timestamps.shape[0]
        with tf.Session(graph=simulation_graph) as sess:
            # Initialize the persistent variables
            sess.run(init)
            # Calculate and store the initial state and initial observation
            initial_time, initial_state, initial_observation = sess.run([time, state, observation])
            state_trajectory = _initialize_trajectory(
                num_timestamps,
                1,
                self.state_structure,
                initial_state
            )
            observation_trajectory = _initialize_trajectory(
                num_timestamps,
                1,
                self.observation_structure,
                initial_observation
            )
            # Calculate and store the state and observation for all subsequent time steps
            for timestamp_index in range(1, num_timestamps):
                next_time, next_state, next_observation = sess.run([assign_time, assign_state, assign_observation])
                state_trajectory = _extend_trajectory(
                    state_trajectory,
                    timestamp_index,
                    self.state_structure,
                    next_state
                )
                observation_trajectory = _extend_trajectory(
                    observation_trajectory,
                    timestamp_index,
                    self.observation_structure,
                    next_observation
                )
        return state_trajectory, observation_trajectory

    def estimate_state_trajectory(
        self,
        num_particles,
        timestamps,
        observation_trajectory):
        # Build the dataflow graph
        state_trajectory_estimation_graph = tf.Graph()
        with state_trajectory_estimation_graph.as_default():
            # Sample the global parameters
            parameters = self.parameter_model_sample()
            # Calculate the initial values for the persistent variables
            timestamps_iterator = tf.data.Dataset.from_tensor_slices(timestamps).make_one_shot_iterator()
            observation_trajectory_iterator_dict = _make_iterator_dict(
                self.observation_structure,
                observation_trajectory
            )
            initial_time = timestamps_iterator.get_next()
            initial_state = self.initial_model_sample(
                num_particles,
                parameters
            )
            initial_observation = _iterator_dict_get_next(
                self.observation_structure,
                observation_trajectory_iterator_dict
            )
            initial_log_weights = self.observation_model_pdf(
                initial_state,
                initial_observation,
                parameters)
            # Define the persistent variables
            time = tf.get_variable(
                name = 'time',
                dtype = tf.float32,
                initializer = initial_time
            )
            state = _get_variable_dict(self.state_structure, initial_state)
            log_weights = tf.get_variable(
                name='log_weights',
                dtype = tf.float32,
                initializer = initial_log_weights
            )
            # Initialize the persistent variables
            init = tf.global_variables_initializer()
            # Calculate the state samples and log weights for the next time step
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
            next_time = timestamps_iterator.get_next()
            next_state = self.transition_model_sample(
                state_resampled,
                time,
                next_time,
                parameters
            )
            next_observation = _iterator_dict_get_next(
                self.observation_structure,
                observation_trajectory_iterator_dict)
            next_log_weights = self.observation_model_pdf(
                next_state,
                next_observation,
                parameters)
            # Assign these values to the persistent variables so they become the inputs for the next time step
            control_dependencies = [next_time, next_log_weights] + _tensor_list(next_state)
            with tf.control_dependencies(control_dependencies):
                assign_time = time.assign(next_time)
                assign_state = _variable_dict_assign(
                    self.state_structure,
                    state,
                    next_state
                )
                assign_log_weights = log_weights.assign(next_log_weights)
        # Run the calcuations using the graph above
        num_timestamps = timestamps.shape[0]
        with tf.Session(graph=state_trajectory_estimation_graph) as sess:
            # Initialize the persistent variables
            sess.run(init)
            # Calculate and store the initial state samples and log weights
            initial_time, initial_state, initial_log_weights = sess.run([time, state, log_weights])
            state_trajectory = _initialize_trajectory(
                num_timestamps,
                num_particles,
                self.state_structure,
                initial_state
            )
            log_weights_trajectory = np.zeros((num_timestamps, num_particles))
            log_weights_trajectory[0] = initial_log_weights
            # Calculate and store the state samples and log weights for all subsequent time steps
            for timestamp_index in range(1, num_timestamps):
                next_time, next_state, next_log_weights = sess.run([assign_time, assign_state, assign_log_weights])
                state_trajectory = _extend_trajectory(
                    state_trajectory,
                    timestamp_index,
                    self.state_structure,
                    next_state
                )
                log_weights_trajectory[timestamp_index] = next_log_weights
        return state_trajectory, log_weights_trajectory

def _get_variable_dict(structure, initial_values):
    variable_dict = {}
    for variable_name, variable_info in structure.items():
        variable_dict[variable_name] = tf.get_variable(
            name = variable_name,
            dtype = variable_info['dtype'],
            initializer = initial_values[variable_name])
    return variable_dict

def _variable_dict_assign(structure, variable_dict, values):
    assign_dict = {}
    for variable_name in structure.keys():
        assign_dict[variable_name] = variable_dict[variable_name].assign(values[variable_name])
    return assign_dict

def _make_iterator_dict(structure, array_dict):
    iterator_dict={}
    for variable_name in structure.keys():
        dataset = tf.data.Dataset.from_tensor_slices(
            np.asarray(
                array_dict[variable_name],
                dtype=np.float32
            )
        )
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
