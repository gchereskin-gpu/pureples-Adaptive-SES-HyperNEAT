

This repo is home to a novel extension of Adaptive ES-HyperNEAT called Adaptive Specialized-ES-HyperNEAT. The following abstract overviews Adaptive SES-HyperNEAT's novelty:
=================================================================================================================================================================================


  Neuroevolution mimics Darwinian evolution to evolve AI neural networks from scratch for control problems such as robotic control. Many features of biological brains have yet to be implemented in neuroevolution algorithms and may improve their ability to evolve intelligence.
  
  One such feature is the evolution of functionally distinct neuron types fundamental to biological brains’ intricate circuitry. Prior neuroevolution algorithms encode the circuitry and plasticity (adaptive characteristics) of neural networks as a geometric pattern, but this does not provide the discrete functional organization underlying biological motifs, neuroplasticity, neuromodulation, topography, etc. Therefore, this study introduces a novel extension of Adaptive ES-HyperNEAT called Adaptive Specialized-ES-HyperNEAT which evolves multiple ‘branches’ of connection weight output nodes in the encoding and evolves sets of Hebbian coefficients for respective branches’ connections to share, mirroring the organized functional diversity of biological brains.
  
  Comparing the performance of Adaptive SES-HyperNEAT to Adaptive ES-HyperNEAT on a continuous T-maze task, Adaptive SES-HyperNEAT demonstrated more exploration and generalization but less exploitation. This may have been due to the algorithm’s less precise encoding of plasticity (multiple connections shared the same Hebbian coefficients). However, the novel algorithm’s significantly increased speciation sheds light on the fragility of neuroevolution’s results under the unideal manual tuning of parameters. Provided that this is improved for further development of NEAT’s extensions, Adaptive SES-HyperNEAT opens the door to implementing significant features of biological neural networks such as functionally-specific global neuromodulation, advancing our understanding of how neuroevolution algorithms can be improved with future inspiration from biology for evolving more intelligent neural networks.



Included in this repo is my custom cat-and-mouse domain for a personal project and helpful for further testing of Adaptive SES-HyperNEAT.
=
Cat-and-mouse was not used in the aforementioned study, but to evolve adaptive NNs for controlling my "RoboSphere" - an autonomous spherical rolling robot designed to actively evade and play with my dog.

The following is a section of entries from my RoboSphere design journal, tracking some of my thought process of coding the cat-and-mouse domain.
-

Jul 23 (2026)

Plan:
Make design requirements
No solutions, just requirements

The hamster wheel will be best, but I need a separate isolated pendulum for the sensors
Making the dynamics of kinematics in an environment is important


Jul 29

Plan:
Figure out specifics of hamster design
Finish design requirements (first take)
I want to use a three-wheel design & a circular arrangement of sensors


Aug 12

Plan:
complete preliminary parts list & measurements
Half did parts, half did 3D model -> they inform each-other


Aug 14
	Made lots of progress with the code (implemented velocities/momentum, rings of input nodes, etc.). Big problem is that the rotational velocity is just making the agents spin forever.


Aug 15
	Had a great idea - I was going to have the output nodes control pairs of 2 motors so that outputs of each node make it go straight, but the three outputs would still have to be balanced, so instead, I made a ring of output nodes in the substrate which control the direction of movement. This complements the nonholonomic movement & mapping of inputs much better. Now evolution is evolving some solutions


Aug 17
-
Thought of RoboSphere from a different perspective and considered some important questions. As a ‘product’, it needs to be easy to use. I should be able to remotely turn it on/off (perhaps on the dog harness). It could even have different modes, such as selecting different neural networks evolved with different fitness functions, a mode that stops and starts playing periodically to engage with Thor and let him rest.

Aug 18

Faced some new challenges with moving onto the building process. I did some math (paper and Mathematica) to figure out dimensions of the body angles for the motors and the offset from the equator of the sphere at which the motors should sit. It worked out beautifully which was cool. But then I had a hard time finding the right omniwheels and motors. I researched what motors I need - options and such. And many of the motors that would work are not the right size. One was close but is just a little too big. Not sure what to do… It’s challenging, too, that so many parts work together, because the size of the wheels affect the motors and the adapters, and the motors affect the battery, and the battery affects the weight, and the motors affect the encoder, etc.

Sep 26

Email BD and got a very insightful reply about the design process, and I want to apply it here. COTS components are the way to go right now, and if I can’t find motors that fit properly, I am going to take a new approach and make the sphere bigger. There aren’t particularly any downsides to this, and it gives me an opportunity to buy stronger and better spheres for the sensors to see through.


Oct: Rest of the CAD design process not recorded in Github (it’s unrelated to the code)
-


Below is the description of Pureples:
-
PUREPLES - Pure Python Library for ES-HyperNEAT
===============================================

About
-----
This is a library of evolutionary algorithms with a focus on neuroevolution, implemented in pure python, depending on the [neat-python](https://github.com/CodeReclaimers/neat-python) implementation. It contains a faithful implementation of both HyperNEAT and ES-HyperNEAT which are briefly described below.

**NEAT** (NeuroEvolution of Augmenting Topologies) is a method developed by Kenneth O. Stanley for evolving arbitrary neural networks.  
**HyperNEAT** (Hypercube-based NEAT) is a method developed by Kenneth O. Stanley utilizing NEAT. It is a technique for evolving large-scale neural networks using the geometric regularities of the task domain.  
**ES-HyperNEAT** (Evolvable-substrate HyperNEAT) is a method developed by Sebastian Risi and Kenneth O. Stanley utilizing HyperNEAT. It is a technique for evolving large-scale neural networks using the geometric regularities of the task domain. In contrast to HyperNEAT, the substrate used during evolution is able to evolve. This rids the user of some initial work and often creates a more suitable substrate.

The library is extensible in regards to easy transition between experimental domains.

Getting started
---------------
This section briefly describes how to install and run experiments.  

### Installation Guide
First, make sure you have the dependencies installed: `numpy`, `neat-python`, `graphviz`, `matplotlib` and `gym`.  
All the above can be installed using [pip](https://pip.pypa.io/en/stable/installing/).  
Next, download the source code and run `setup.py` (`pip install .`) from the root folder. Now you're able to use **PUREPLES**!

### Experimenting
How to experiment using NEAT will not be described, since this is the responsibility of the `neat-python` library.

Setting up an experiment for **HyperNEAT**:
* Define a substrate with input nodes and output nodes as a list of tuples. The hidden nodes is a list of lists of tuples where the inner lists represent layers. The first list is the topmost layer, the last the bottommost.
* Create a configuration file defining various NEAT specific parameters which are used for the CPPN.
* Define a fitness function setting the fitness of each genome. This is where the CPPN and the ANN is constructed for each generation - use the `create_phenotype_network` method from the `hyperneat` module.
* Create a population with the configuration file made in (2).
* Run the population with the fitness function made in (3) and the configuration file made in (2). The output is the genome solving the task or the one closest to solving it.

Setting up an experiment for **ES-HyperNEAT**:
Use the same setup as HyperNEAT except for:
* Not declaring hidden nodes when defining the substrate.
* Declaring ES-HyperNEAT specific parameters.
* Using the `create_phenotype_network` method residing in the `es_hyperneat` module when creating the ANN.

If one is trying to solve an experiment defined by the [OpenAI Gym](https://gym.openai.com/) it is even easier to experiment. In the `shared` module a file called `gym_runner` is able to do most of the work. Given the number of generations, the environment to run, a configuration file, and a substrate, the relevant runner will take care of everything regarding population, fitness function etc.

Please refer to the sample experiments included for further details on experimenting. 

