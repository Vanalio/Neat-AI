# NEAT

## FIXME

elite transfer must go before weak extinction
parent selection must be based on chance proportional to rank of genome within the species, not chance proportional to its fitness

## ADD

randomize activation function at creation or mutate_add_neuron with random.choice(ActivationFunctions.get_activation_functions())
implement update of age and generations without improvement
implement interspecies mating
implement parallel evaluation

## CHECK

check if the network is built correctly and the computation is correct
disabled genes are correctly inherited
removals should remove something more than just what they remove (dependencies?)
purge redundant methods

## ChatGPT

### Problems

consider this network:

input neurons 1, 2 ,3 and 4
hidden neurons 7, 8, and 9
output neurons 5 and 6

input sequence
first timestep [1, 2, 3, 4]
second timestep [5, 6, 7 ,8]

all input neurons no activation no bias

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

as you can see the hidden layer presents recurrence, with bidirectional connections, self connections and loops.

write the python code that will:

give network output at each timestep
considering loops, self connections, bidirectional connections
you need to somehow manage the rnn using state saving

describe the flow of the data moving among each neuron so that its clear how the data moves from input to output across the network.
