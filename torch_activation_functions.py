import torch
import numpy as np

from config import Config

config = Config("config.ini", "DEFAULT")


class ActivationFunctions:
    @staticmethod
    def brain_functions():
        return ["brain_relu", "brain_tanh", "brain_softsign", "multi_brain_relu", "multi_brain_tanh", "multi_brain_softsign"]

    @staticmethod
    def bipolar_functions():
        return ["tanh", "bipolar_sigmoid", "softsign", "bipolar_arctan"]

    @staticmethod
    def brain_sigmoid(x):
        return torch.where(x >= 0, 2 * torch.sigmoid(x) - 1, torch.zeros_like(x))

    @staticmethod
    def brain_relu(x, relu_clip_at=config.relu_clip_at):
        return torch.clamp(x, min=0, max=relu_clip_at)
    
    @staticmethod
    def brain_tanh(x):
        return torch.where(x >= 0, torch.tanh(x), torch.zeros_like(x))

    @staticmethod
    def brain_softsign(x):
        return torch.where(x >= 0, x / (1 + torch.abs(x)), torch.zeros_like(x))

    @staticmethod
    def brain_arctan(x):
        return torch.where(x >= 0, 2 * (torch.atan(x) / torch.tensor(np.pi)), torch.zeros_like(x))

    @staticmethod
    def multi_brain_sigmoid(x):
        return torch.where(x > 5, torch.zeros_like(x),
                           torch.where(x > 1, ActivationFunctions.brain_sigmoid(x - 3),
                                       ActivationFunctions.brain_sigmoid(x)))

    @staticmethod
    def multi_brain_relu(x, relu_clip_at=config.relu_clip_at):
        return torch.where(x > 5, torch.zeros_like(x),
                           torch.where(x > 1, ActivationFunctions.brain_relu(x - 3, relu_clip_at),
                                       ActivationFunctions.brain_relu(x, relu_clip_at)))

    @staticmethod
    def multi_brain_tanh(x):
        return torch.where(x > 5, torch.zeros_like(x),
                           torch.where(x > 1, ActivationFunctions.brain_tanh(x - 3),
                                       ActivationFunctions.brain_tanh(x)))

    @staticmethod
    def multi_brain_softsign(x):
        return torch.where(x > 5, torch.zeros_like(x),
                           torch.where(x > 1, ActivationFunctions.brain_softsign(x - 3),
                                       ActivationFunctions.brain_softsign(x)))

    @staticmethod
    def multi_brain_arctan(x):
        return torch.where(x > 5, torch.zeros_like(x),
                           torch.where(x > 1, ActivationFunctions.brain_arctan(x - 3),
                                       ActivationFunctions.brain_arctan(x)))

    @staticmethod
    def relu(x):
        return torch.relu(x)

    @staticmethod
    def leaky_relu(x):
        return torch.nn.functional.leaky_relu(x)

    @staticmethod
    def tanh(x):
        return torch.tanh(x)

    @staticmethod
    def sigmoid(x):
        return torch.sigmoid(x)

    @staticmethod
    def bipolar_sigmoid(x):
        return 2 * torch.sigmoid(x) - 1

    @staticmethod
    def softplus(x):
        return torch.nn.functional.softplus(x)

    @staticmethod
    def abs(x):
        return torch.abs(x)

    @staticmethod
    def softsign(x):
        return x / (1 + torch.abs(x))

    @staticmethod
    def arctan(x):
        return torch.atan(x)

    @staticmethod    
    def bipolar_arctan(x):
        return 2 * torch.atan(x) / torch.tensor(np.pi)

    @staticmethod
    def identity(x):
        return x
