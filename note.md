# NEAT

## FIXME

- speciation is broken
- move render genome to visualizer, then fix the entire visualizer (included load from file)
- gpu is slower

## IMPROVE

- refactor from_neuron, to neuron --> from_neuron_id, to_neuron_id
- data types

## ADD

- progress bars
- use __str__
- add connections can select a "from layer to layer" pair already full when an other "from layer to layer" pair is not
- implement interspecies mating

## CHECK

- remove redundant returns when in functions with side effects if not used anywhere
- adding other useful values to the ones already returned by methods
- redundant computing of things already present as attributes in self or elsewhere
