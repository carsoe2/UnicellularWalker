# Determining an optimal walking pattern for a unicellular walker
## Run Instructions
- pip install numpy matplotlib
- python noisy_walker_mcts.py
- Output file: simulation_results.txt
## Background
This is a research project which I will begin working on with Professor Agung Julius (RPI) and Ben Larson (Berkeley) next semester. Larson previously published a paper studying a unicellular organism called Euplotes eurystomus which has 14 leg-like appendages called cirri. The goal of this paper was to determine if there was a pattern in the way the cirri moved such that the organism can navigate its environment. Unlike highly complex multicellular organisms, unicellular organisms do not have a central nervous system which allows coordination in the movement in appendages. Larson’s publication on this topic can be found [here](https://doi.org/10.1016/j.cub.2022.07.034).
## Goal
Since this publication, a simulator for the organism’s gait has been developed. It was found that microtubules within the organism determined leg movement, and the positions of the cirri could be determined by a finite-state machine. In preparation for exploring how I can contribute to this research project next semester, a question which could be explored in the context of this class is if there is a pattern of cirri movement which results in optimal movement. That is, given the simulator and the knowledge that cirri movement is representable by a finite-state machine, determine the optimal sequence which maximizes the distance which Euplotes eurystomus can travel.
## Methodology
Monte Carlo Tree search was be the learning method used since the number of iterations would be equivalent to the depth parameter and good to use for exploring different cirri movements (exploration vs exploitation).
