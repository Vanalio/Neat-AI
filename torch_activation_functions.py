# Activation functions used in the neural network


import torch


class ActivationFunctions:
    @staticmethod
    def get_activation_functions():
        return [
            "identity",
            "relu",
            #"leaky_relu",
            "clipped_relu",
            "tanh",
            #"sigmoid",
            #"softplus",
            #"abs",
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