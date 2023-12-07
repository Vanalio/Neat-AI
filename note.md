# NEAT

## FIXME

- BEST GENOME: 253932, Fitness: 47228.408363353024, connections: 18, hidden neurons: 1 ???
- add connections is broken
- refactor from_neuron, to neuron to from_neuron_id, to_neuron_id
- move render genome to visualizer
- fix the entire visualizer (included load from file)

## ADD

- use __str__
- more details on best genome printout (activation functions and biases)

- option to mutate or not and to initialize to fixed or random:
    default_hidden_activation  = random
    mutate_hidden_activation   = True
    default_output_activation  = identity
    mutate_output_activation   = False

- add connections can select a "from layer to layer" couple already full when an other "from layer to layer" couple is not
- implement generations without improvement
- implement interspecies mating

## CHECK

- ensure that the data types (like float32, float64) are consistent
- remove redundant returns when in functions with side effects if not used anywhere
- adding other values to the ones already returned if useful
- redundant computing of things already present as attributes in self or elsewhere
