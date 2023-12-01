# NEAT

## FIXME

- move render genome to visualize class and fix it

## CHECK

- all the returns (redundant, mixed to side effects, computing them when not needed because an attribute is present in self or elsewhere, etc.)
- what happens to neurons (copied, cloned). when inherited or muted, shared?

## ADD

- add connections works but can select from layer to layer couple already full when other from layer to layer couple is not
- implement generations without improvement
- implement interspecies mating
- ignore # in the config file
