# NEAT

## FIXME

### speciation is broken
explanation will soon follow :)

### move test genome to visualizer? Fix the visualizer
- move the geneome test run (render_mode=human) in the visualizer?
- fitness plot and genome structure figs are not synchronized and i think theres an offset of 3 generations from the current one that can be reduced

### Load from file
load from file, for genomes and population seems not to work (doesnt take all the needed obj?)

### gpu is slower
if i move to gpu everything is slower, maybe to much overhead moving stuff to memory or im moving data here and there too much

## IMPROVE

### refactor from_neuron, to neuron --> from_neuron_id, to_neuron_id
from_neuron to_neuron in connectiongenes are id and I would like the name of those attributes to be named accorodingly to avoid confusion with the real object.

### use list instead of dict where id and not entire object
On a similar note there are places where I think I don't need a dictonary of objects but the list of their id would just be enough. ie Population and Species have genomes as entire objects but I could just reference their id.

### data types
I would like to explicity declare expected data types as per best practice

## ADD

### progress bars
add progress bars everywhere possible

### use __str__
define the __str__ methods where useful instead of using all those prints

### add connections can select a "from layer to layer" pair already full when an other "from layer to layer" pair is not
When a connection is added 

### implement interspecies mating
this is easy but not so important now

## CHECK

### remove redundant returns when in functions with side effects if not used anywhere
if a func has side effects i should remove the returns if not needed

### adding other useful values to the ones already returned by methods
maybe i can add other values to some returns that can be used elsewhere instead of computing those values again with another method?

### redundant computing of things already present as attributes in self or elsewhere
need to check if i compute something that i already have (ie I compute number of input neurons when instead are already listed in an attribute)
