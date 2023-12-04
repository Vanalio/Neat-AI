# NEAT

## FIXME

- move render genome to visualizer  and fix the load from file

## ADD

- use __str__
- batch processing
- more details on best genome printout (activation functions and biases)

- option to mutate or not and to initialize to fixed or random:
    default_hidden_activation  = random
    mutate_hidden_activation   = True
    default_output_activation  = identity
    mutate_output_activation   = False

- add connections works but can select from layer to layer couple already full when other from layer to layer couple is not
- implement generations without improvement
- implement interspecies mating

## CHECK

- remove redundant returns because mixed to side effects and not used elsewhere
- add returns where useful
- redundant computing of things already present as attributes in self or elsewhere
