# Activation functions used in the neural network


import torch


class ActivationFunctions:
    @staticmethod
    def get_activation_functions():
        return [
            #"identity",
            #"abs",
            #"softplus",
            #"relu",
            #"leaky_relu",
            "clipped_relu",
            #"sigmoid",
            "brain_sigmoid",
            "softsign"
            #"tanh",
            #"arctan"
        ]

    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def relu(x):
        return torch.relu(x)

    @staticmethod
    def leaky_relu(x):
        return torch.nn.functional.leaky_relu(x)

    @staticmethod
    def clipped_relu(x, relu_clip_at=1):
        return torch.clamp(x, min=0, max=relu_clip_at)

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
    def brain_sigmoid(x):
        return torch.where(x >= 0, 2 * torch.sigmoid(x) - 1, torch.zeros_like(x))

    @staticmethod
    def softsign(x):
        return x / (1 + torch.abs(x))

    @staticmethod
    def arctan(x):
        return torch.atan(x)