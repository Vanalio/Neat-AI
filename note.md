# FIXME
elite transfer must go before weak extinction
parent selection must be based on chance proportional to rank of genome within the species, not chance proportional to its fitness

# ADD
randomize activation function at creation or mutate_add_neuron with random.choice(ActivationFunctions.get_activation_functions())
implement update of age and generations without improvement
implement interspecies mating
implement parallel evaluation

# CHECK
check if the network is built correctly and the computation is correct
disabled genes are correctly inherited
removals should remove something more than just what they remove (dependencies?)
purge redundant methods
