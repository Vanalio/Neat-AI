FIXME: cull and stale species seem not to work properly

FIXME: crossover count is not correct

FIXME: implement parallel evaluation

FIXME: parent selection must be based on chance proportional to rank of genome within the species, not chance proportional to its fitness

FIXME: implement update of age and generations without improvement

FIXME: implement interspecies mating

ADD: randomize activation function at creation or mutate_add_neuron with random.choice(ActivationFunctions.get_activation_functions())

CHECK: check if the network is built correctly and the computation is correct

CHECK: disabled genes are correctly inherited 

CHECK: removals should remove something more than just what they remove (dependencies?)

CHECK: purge redundant methods