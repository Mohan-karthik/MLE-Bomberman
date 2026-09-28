import numpy as np


def new_model(number_of_actions, number_of_features):
    return {
        'weights': np.zeros((number_of_actions, number_of_features), dtype=np.float32),
        'episodes': 0,
        'epsilon': 1.0,
    }


def q_values(model, features):
    return model['weights'] @ features


def update_q(model, action_index, features, target, alpha):
    prediction = float(model['weights'][action_index] @ features)
    td_error = target - prediction
    model['weights'][action_index] += alpha * td_error * features
    return td_error
