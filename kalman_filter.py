import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter(object):
    def __init__(
        self,
        state_transition=None,
        control_input=None,
        observation=None,
        process_noise=None,
        observation_noise=None,
        start_covariance_matrix=None,
        start_state=None,
    ):
        if state_transition is None or observation is None:
            raise ValueError("Set proper system dynamics.")

        self.n = state_transition.shape[1]
        self.m = observation.shape[1]

        self.state_transition = state_transition
        self.observation = observation
        self.control_input = 0 if control_input is None else control_input
        self.process_noise = np.eye(self.n) if process_noise is None else process_noise
        self.observation_noise = (
            np.eye(self.n) if observation_noise is None else observation_noise
        )
        self.current_coverage_matrix = np.eye(self.n) if start_covariance_matrix is None else start_covariance_matrix
        self.current_state = np.zeros((self.n, 1)) if start_state is None else start_state

    def predict(self, u=0):
        self.current_state = np.dot(self.state_transition, self.current_state) + np.dot(
            self.control_input, u
        )
        self.current_coverage_matrix = (
            np.dot(
                np.dot(self.state_transition, self.current_coverage_matrix),
                self.state_transition.T,
            )
            + self.process_noise
        )
        return self.current_state

    def update(self, measurement):
        y = measurement - np.dot(self.observation, self.current_state)
        S = self.observation_noise + np.dot(
            self.observation, np.dot(self.current_coverage_matrix, self.observation.T)
        )
        kalman_gain = np.dot(
            np.dot(self.current_coverage_matrix, self.observation.T), np.linalg.inv(S)
        )
        self.current_state = self.current_state + np.dot(kalman_gain, y)
        identity_matrix = np.eye(self.n)
        self.current_coverage_matrix = (
            np.dot(
                np.dot(identity_matrix - np.dot(kalman_gain, self.observation), self.current_coverage_matrix),
                (identity_matrix - np.dot(kalman_gain, self.observation)).T,
            )
            + np.dot(np.dot(kalman_gain, self.observation_noise), kalman_gain.T)
        )


def example():
    dt = 1.0 / 60
    state_transition = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
    observation = np.array([1, 0, 0]).reshape(1, 3)
    process_noise = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
    observation_noise = np.array([0.5]).reshape(1, 1)

    x = np.linspace(-10, 10, 100)
    measurements_over_time = -(x ** 2 + 2 * x - 2) + np.random.normal(0, 2, 100)

    kalman_filter = KalmanFilter(
        state_transition=state_transition,
        observation=observation,
        process_noise=process_noise,
        observation_noise=observation_noise
    )
    predictions = []

    for measurement in measurements_over_time:
        predictions.append(np.dot(observation, kalman_filter.predict())[0])
        kalman_filter.update(measurement)

    plt.plot(range(len(measurements_over_time)), measurements_over_time, label="Measurements")
    plt.plot(
        range(len(predictions)), np.array(predictions), label="Kalman Filter Prediction"
    )
    plt.legend()
    plt.show()
