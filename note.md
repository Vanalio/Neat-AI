# NEAT

## FIXME

- remove neuron
- add connections is possibly broken
- refactor from_neuron, to neuron to from_neuron_id, to_neuron_id
- move render genome to visualizer, then fix the entire visualizer (included load from file)

## ADD

- use __str__

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
