# SNS: SNS’s not a Synthesizer: A Deep-Learning-Based Synthesis
## Abstract
The number of transistors that can fit on one monolithic chip has reached billions to tens of billions in this decade thanks to Moore’s Law. With the advancement of every technology generation, the transistor counts per chip grow at a pace that brings about exponen- tial increase in design time, including the synthesis process used to perform design space explorations. Such a long delay in obtaining synthesis results hinders an efficient chip development process, sig- nificantly impacting time-to-market. In addition, these large-scale integrated circuits tend to have larger and higher-dimension design spaces to explore, making it prohibitively expensive to obtain physi- cal characteristics of all possible designs using traditional synthesis tools.

In this work, we propose a deep-learning-based synthesis pre- dictor called SNS (SNS’s not a Synthesizer), that predicts the area, power, and timing physical characteristics of a broad range of de- signs at two to three orders of magnitude faster than the Synopsys Design Compiler while providing on average a 0.4998 RRSE (root relative square error). We further evaluate SNS via two representa- tive case studies, a general-purpose out-of-order CPU case study using RISC-V Boom open-source design and an accelerator case study using an in-house Chisel implementation of DianNao, to demonstrate the capabilities and validity of SNS.

## Code Structure


## Run
### Setup Dependency
* Anaconda/Miniconda
* Python 3.8+ (Tested Version: Python `3.9.12`)
* Pip
* All python modules listed in `requirements.txt`.
* Yosys (Tested Version: `Yosys 0.9+3752 (git sha1 c4645222, gcc 9.3.0-17ubuntu1~20.04 -fPIC -Os)`)
* Anything that submodule `random-ff2ff-gen` requires. 
* Anything that is required to make `cuda:0` device available in PyTorch environment.

### Run
### Training Custom Model

### Generating Custom Dataset
*TODO*
Please watch our new publication at [Apex Lab](https://apexlab-duke.github.io) for the published version of the dataset.

## Citation
```
@inproceedings{10.1145/3470496.3527444,
author = {Xu, Ceyu and Kjellqvist, Chris and Wills, Lisa Wu},
title = {SNS's Not a Synthesizer: A Deep-Learning-Based Synthesis Predictor},
year = {2022},
isbn = {9781450386104},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3470496.3527444},
doi = {10.1145/3470496.3527444},
abstract = {The number of transistors that can fit on one monolithic chip has reached billions to tens of billions in this decade thanks to Moore's Law. With the advancement of every technology generation, the transistor counts per chip grow at a pace that brings about exponential increase in design time, including the synthesis process used to perform design space explorations. Such a long delay in obtaining synthesis results hinders an efficient chip development process, significantly impacting time-to-market. In addition, these large-scale integrated circuits tend to have larger and higher-dimension design spaces to explore, making it prohibitively expensive to obtain physical characteristics of all possible designs using traditional synthesis tools.In this work, we propose a deep-learning-based synthesis predictor called SNS (SNS's not a Synthesizer), that predicts the area, power, and timing physical characteristics of a broad range of designs at two to three orders of magnitude faster than the Synopsys Design Compiler while providing on average a 0.4998 RRSE (root relative square error). We further evaluate SNS via two representative case studies, a general-purpose out-of-order CPU case study using RISC-V Boom open-source design and an accelerator case study using an in-house Chisel implementation of DianNao, to demonstrate the capabilities and validity of SNS.},
booktitle = {Proceedings of the 49th Annual International Symposium on Computer Architecture},
pages = {847–859},
numpages = {13},
keywords = {neural networks, RTL-level synthesis, integrated circuits},
location = {New York, New York},
series = {ISCA '22}
}
```

