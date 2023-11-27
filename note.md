# NEAT

## FIXME

action return values out of range?
add connections, max_new, attempts not working?

## ADD

cuda gpu device
randomize activation function at creation or mutate_add_neuron with random.choice(ActivationFunctions.get_activation_functions())
implement generations without improvement
implement interspecies mating
implement parallel evaluation
ignore # in the config file

## CHECK

check if the network is built correctly and the computation is correct
disabled genes are correctly inherited
mutate add connections, attempts, max tries, etc

## ChatGPT

### Problems

consider this network:

input neurons 1, 2 ,3 and 4
hidden neurons 7, 8, and 9
output neurons 5 and 6

input sequence
1st timestep [1, -2, 3, -4]
2nd timestep [-5, 6, -7, 8]
3rd timestep [-1, 2, -3, 4]
4th timestep [1, -2, 3, -4]
5th timestep [-5, 6, -7, 8]
6th timestep [-1, 2, -3, 4]

all input neurons have no activation and no bias
all hidden neuron activation: relu clipped at 1
all hidden neuron bias: 10
all output neuron activation: softsign
all output neuron bias: 0

connections (all weights 1.5)
1 to 7
2 to 7 and 9
3 to 8
4 to 9
7 to 5 and 9
9 to 7 and 6
8 to 7, 8 and 6

as you can see the hidden layer presents recurrence, with a) bidirectional connections, b) self connections and c) loops.

write the python code that considering loops, self connections and bidirectional connections will give the network output at each timestep

you need to somehow manage the rnn using state saving.

use the best approach in terms of speed and memory footprint. we have access to gpus if needed.

the code must be VERY short and still completely functional!!!

make a class for the neurons with: neuron_id, layer, activation, bias
make a class for connections with: from_neuron_id, to neuron_id, weight

build a network that manage all the computations and fed with an input gives output at each timestep

it needs to be able to manage the most general case, the most various recurrent topologies

use pytorch at each and every possible point
