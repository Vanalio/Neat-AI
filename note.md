# NEAT

## FIXME

- mutate, reproduction or next gen broken
- move render genome to visualize class and fix it

## CHECK

- all the returns (redundant, mixed to side effects, computing them when not needed because an attribute is present in self or elsewhere, etc.)

## ADD

- option to mutate or not and to initialize to fixed or random):
    default_hidden_activation  = random
    mutate_hidden_activation   = True
    default_output_activation  = identity
    mutate_output_activation   = False

- add connections works but can select from layer to layer couple already full when other from layer to layer couple is not
- implement generations without improvement
- implement interspecies mating
- ignore # in the config file
