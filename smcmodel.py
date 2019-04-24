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

    def simulate(self, timestamps):
        simulation_graph = tf.Graph()
        with simulation_graph.as_default():
            parameters = self.parameter_model_sample()
            initial_time = tf.constant(timestamps[0], dtype=tf.float32)
            initial_state = self.initial_model_sample(1, parameters)
            initial_observation = self.observation_model_sample(initial_state, parameters)
            time = tf.get_variable(
                name = 'time',
                dtype = tf.float32,
                initializer = initial_time)
            state = _define_variables(self.state_structure, initial_state)
            observation = _define_variables(self.observation_structure, initial_observation)
            init = tf.global_variables_initializer()
            timestamps_dataset = tf.data.Dataset.from_tensor_slices(timestamps[1:])
            timestamps_iterator = timestamps_dataset.make_one_shot_iterator()
            current_time = time
            current_state = state
            current_observation = observation
            next_time = timestamps_iterator.get_next()
            next_state = self.transition_model_sample(
                current_state,
                current_time,
                next_time,
                parameters)
            next_observation = self.observation_model_sample(next_state, parameters)
            control_dependencies = [next_time] + _tensor_list(next_state) + _tensor_list(next_observation)
            with tf.control_dependencies(control_dependencies):
                assign_time = time.assign(next_time)
                assign_state = _assign_variables(
                    self.state_structure,
                    state,
                    next_state
                )
                assign_observation = _assign_variables(
                    self.observation_structure,
                    observation,
                    next_observation
                )
        num_timestamps = timestamps.shape[0]
        with tf.Session(graph=simulation_graph) as sess:
            sess.run(init)
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
        state_trajectory_estimation_graph = tf.Graph()
        with state_trajectory_estimation_graph.as_default():
            parameters = self.parameter_model_sample()
            initial_time = tf.constant(timestamps[0], dtype=tf.float32)
            observation_trajectory_iterators = _define_iterators(
                self.observation_structure,
                observation_trajectory
            )
            initial_state = self.initial_model_sample(
                num_particles,
                parameters
            )
            initial_observation = _next_value(
                self.observation_structure,
                observation_trajectory_iterators
            )
            initial_log_weights = self.observation_model_pdf(
                initial_state,
                initial_observation,
                parameters)
            time = tf.get_variable(
                name = 'time',
                dtype = tf.float32,
                initializer = initial_time
            )
            state = _define_variables(self.state_structure, initial_state)
            log_weights = tf.get_variable(
                name='log_weights',
                dtype = tf.float32,
                initializer = initial_log_weights
            )
            init = tf.global_variables_initializer()
            timestamps_dataset = tf.data.Dataset.from_tensor_slices(timestamps[1:])
            timestamps_iterator = timestamps_dataset.make_one_shot_iterator()
            observation_trajectory_iterators = _define_iterators(
                self.observation_structure,
                observation_trajectory
            )
            current_time = time
            current_state = state
            current_log_weights = log_weights
            next_time = timestamps_iterator.get_next()
            next_observation = _next_value(
                self.observation_structure,
                observation_trajectory_iterators)
            resample_indices = tf.squeeze(
                tf.random.categorical(
                    [log_weights],
                    num_particles
                )
            )
            current_state_resampled = _resample_tensor_dict(
                self.state_structure,
                current_state,
                resample_indices)
            next_state = self.transition_model_sample(
                current_state_resampled,
                current_time,
                next_time,
                parameters
            )
            next_log_weights = self.observation_model_pdf(
                next_state,
                next_observation,
                parameters)
            control_dependencies = [next_time, next_log_weights] + _tensor_list(next_state)
            with tf.control_dependencies(control_dependencies):
                assign_time = time.assign(next_time)
                assign_state = _assign_variables(
                    self.state_structure,
                    state,
                    next_state
                )
                assign_log_weights = log_weights.assign(next_log_weights)
        num_timestamps = timestamps.shape[0]
        with tf.Session(graph=state_trajectory_estimation_graph) as sess:
            sess.run(init)
            initial_time, initial_state, initial_log_weights = sess.run([time, state, log_weights])
            state_trajectory = _initialize_trajectory(
                num_timestamps,
                num_particles,
                self.state_structure,
                initial_state
            )
            log_weights_trajectory = np.zeros((num_timestamps, num_particles))
            log_weights_trajectory[0] = initial_log_weights
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

def _define_variables(structure, initial_values):
    variable_dict = {}
    for variable_name, variable_info in structure.items():
        variable_dict[variable_name] = tf.get_variable(
            name = variable_name,
            dtype = variable_info['dtype'],
            initializer = initial_values[variable_name])
    return variable_dict

def _assign_variables(structure, variables, values):
    assign_dict = {}
    for variable_name in structure.keys():
        assign_dict[variable_name] = variables[variable_name].assign(values[variable_name])
    return assign_dict

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

def _define_iterators(structure, value_trajectory):
    iterators={}
    for variable_name in structure.keys():
        dataset = tf.data.Dataset.from_tensor_slices(
            np.asarray(
                value_trajectory[variable_name],
                dtype=np.float32
            )
        )
        iterators[variable_name] = dataset.make_one_shot_iterator()
    return iterators

def _next_value(structure, iterators):
    next_value = {}
    for variable_name in structure.keys():
        next_value[variable_name] = iterators[variable_name].get_next()
    return next_value

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
