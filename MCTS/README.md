# Determining an optimal walking pattern for a unicellular walker
## Run Instructions
pip install numpy matplotlib
python noisy_walker_mcts.py
Output file: simulation_results.txt
## Background
This is a research project which I will begin working on with Professor Agung Julius (RPI) and Ben Larson (Berkeley) next semester. Larson previously published a paper studying a unicellular organism called Euplotes eurystomus which has 14 leg-like appendages called cirri. The goal of this paper was to determine if there was a pattern in the way the cirri moved such that the organism can navigate its environment. Unlike highly complex multicellular organisms, unicellular organisms do not have a central nervous system which allows coordination in the movement in appendages. Larson’s publication on this topic can be found here: https://rpiexchange-my.sharepoint.com/personal/juliua2_rpi_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fjuliua2%5Frpi%5Fedu%2FDocuments%2FAttachments%2FPIIS0960982222011617%2Epdf&parent=%2Fpersonal%2Fjuliua2%5Frpi%5Fedu%2FDocuments%2FAttachments&ct=1730393611419&or=OWA%2DNT%2DMail&cid=7d757154%2Dba88%2Df668%2De86e%2D825066052952&ga=1
## Goal
Since this publication, a simulator for the organism's gait has been developed. It was found that microtubules within the organism determined leg movement, and the positions of the cirri could be determined by a finite-state machine. In preparation for exploring how I can contribute to this research project next semester, a question which could be explored in the context of this class is if there is a pattern of cirri movement which results in optimal movement. That is, given the simulator and the knowledge that cirri movement is representable by a finite-state machine, determine the optimal sequence which maximizes the distance which Euplotes eurystomus can travel.
## Methodology
Monte Carlo Tree search was be the learning method used since the number of iterations would be equivalent to the depth parameter and good to use for exploring different cirri movements (exploration vs exploitation).
## Final Modifications
A constraint was added to the simulation so that no less than three legs should be attached to the substrate at once. Otherwise, the cell will not be able to maintain contact with the substrate in the real world.