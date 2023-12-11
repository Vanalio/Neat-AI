# NEAT

## FIXME

- refactor from_neuron, to neuron --> from_neuron_id, to_neuron_id
- move render genome to visualizer, then fix the entire visualizer (included load from file)

## ADD

- progress bars
- use __str__
- add connections can select a "from layer to layer" pair already full when an other "from layer to layer" pair is not
- implement interspecies mating

## CHECK

- ensure that the data types (like float32, float64) are consistent
- remove redundant returns when in functions with side effects if not used anywhere
- adding other values to the ones already returned if useful
- redundant computing of things already present as attributes in self or elsewhere
