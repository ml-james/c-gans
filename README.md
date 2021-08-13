### cGANs

#### Introduction

This repository is a direct result of my Masters Physics research project at the *University of Cambridge* entitled:
**Fast Simulation of Particle Physics Detectors using Machine Learning Tools**. Herein lies the final report, and the
code used to generate the containing plots. I shall defer to the `.pdf` to explain the motivation, the aims of the study
and the results.

#### Running the scripts

The scripts should be run within a docker container as I found this the only way to ensure this project could be
platform-agnostic with respect to dependencies. From the root of this repository please run the following commands:

1) `docker build -t c-gans .`
2) `docker run --rm -it -v <output-directory-host-machine>:/out --entrypoint bash c-gans`
3) `python3 c_gans/<python-script-to-run> /out <number-of-epochs-to-train>`