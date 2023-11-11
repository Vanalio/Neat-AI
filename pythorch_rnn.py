import random
import numpy as np
import torch
import torch.nn.functional as F

def clipped_relu(input):
    return torch.clamp(F.relu(input), max=1)

def linear(input):
    return input

def select_activation_function(pool):
    return random.choice(list(pool.values()))

def perception_generator(perception_dimensions, max_timesteps):
    perception = np.empty((0, perception_dimensions))  # Start with an empty array.
    for _ in range(1, max_timesteps + 1):  # Begin with 1 and go up to max_timesteps.
        new_perception = np.random.rand(1, perception_dimensions)  # Create a new timestep.
        perception = np.vstack((perception, new_perception))  # Stack the new timestep below the existing ones.
        yield perception  # Yield the updated perception array.

def process_timestep(input_perception, states, connections, weights, biases, activation_function, last_step=False):
    new_states = {key: val.clone() for key, val in states.items()}

    for src, dest in connections:
        if src.startswith("perception") and dest.startswith("magic"):

            weighted_input = weights[(src, dest)] * input_perception[int(src.split("_")[1])]

            biased_input = weighted_input + biases[dest]

            activated_input = activation_function(biased_input)

            new_states[dest] += activated_input

    for src, dest in connections:
        if src.startswith("magic") and dest.startswith("magic"):

            weighted_state = weights[(src, dest)] * states[src]

            new_states[dest] += weighted_state  

    for state in new_states:
        if state.startswith("magic"):

            new_states[state] = activation_function(new_states[state] + biases[state])

    states.update(new_states)

    action_values = {}
    
    if last_step:
        for src, dest in connections:
            if dest.startswith("action"):
                action_values[dest] = activation_function(weights[(src, dest)] * states[src] + biases[dest])

    return action_values

activation_functions_magic = {
    'relu': F.relu,
    'linear': linear,
    'tanh': torch.tanh,
    'clipped_relu': clipped_relu
}

activation_functions_action = {
    'relu': F.relu,
    'linear': linear,
    'tanh': torch.tanh,
    'clipped_relu': clipped_relu
}

activation_functions = {
    'clipped_relu': clipped_relu
}

perception_dimensions = 1
magic_neurons = 10
action_dimensions = 1

max_timesteps = 10

try_connections = magic_neurons * (magic_neurons + perception_dimensions + action_dimensions)

activation_function = select_activation_function(activation_functions)

weights = {}
biases = {}

for i in range(magic_neurons):
    biases[f"magic_{i}"] = torch.randn(1)
for i in range(action_dimensions):
    biases[f"action_{i}"] = torch.randn(1)

connections = set()
attempt = 0

zero_states = {f"magic_{i}": torch.zeros(1) for i in range(magic_neurons)}

while attempt < try_connections:
    attempt += 1
    print(len(connections))

    src = random.choice(list(zero_states.keys()) + [f"perception_{i}" for i in range(perception_dimensions)])
    if "perception" in src:
        dest = f"magic_{random.randint(0, magic_neurons - 1)}"
    elif "magic" in src:
        dest = random.choice([f"magic_{i}" for i in range(magic_neurons)] + [f"action_{i}" for i in range(action_dimensions)])
    else:
        print("continue")
        continue  

    connection = (src, dest)
    if connection not in connections:
        connections.add(connection)

        weights[connection] = torch.randn(1)

perceptions_generator = perception_generator(perception_dimensions, max_timesteps)

for timestep in range(max_timesteps):
    perception = next(perceptions_generator)
    states = zero_states
    for t in range(max_timesteps):
        actions = process_timestep(perception[t, :], states, connections, weights, biases, activation_function, last_step=(t == max_timesteps - 1))
        print(perception, actions, "\n")