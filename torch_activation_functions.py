import torch

from config import Config

config = Config("config.ini", "DEFAULT")


class ActivationFunctions:
    @staticmethod
    def get_activation_functions():
        return ["brain_relu", "brain_sigmoid", "brain_tanh", "brain_softsign", "brain_arctan"]

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
    def identity(x):
        return x
