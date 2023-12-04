# NEAT

## FIXME

- move render genome to visualizer
- fix the entire visualizer (included load from file)

## ADD

- use __str__
- batch processing
- more details on best genome printout (activation functions and biases)

- option to mutate or not and to initialize to fixed or random:
    default_hidden_activation  = random
    mutate_hidden_activation   = True
    default_output_activation  = identity
    mutate_output_activation   = False

- add connections works but can select a "from layer to layer" couple already full when an other "from layer to layer" couple is not
- implement generations without improvement
- implement interspecies mating

## CHECK

- remove redundant returns when in functions with side effects if not used anywhere
- adding other values to the ones already returned if useful
- redundant computing of things already present as attributes in self or elsewhere
