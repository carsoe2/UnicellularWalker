This project focuses on finding a Reinforcement Learning method that can optimize the walking gait of Euplotes eurystomus organisms. Please check the presentation and report for details on the project background and representation for the RL models.

Check the MCTS and DQN folders for their own individual README files

The project has four main parts:
1) matlab_reference_files contains the original simulation produced by Dr. Ben Larson.
    - noisy_walker_v4 runs the simulation for a specific number of iterations starting from a state where all legs are attached to the substrate. The movements are stochastic and are not guaranteed to switch states if the leg is selected to move. 
    - noisy_walker_v5 was modified to read gait patterns by taking an input of state vectors which provided the model with an initial starting state
    - noisy_walker_v5a modified noisy_walker_v5 to accept another vector which informed the simulation which leg should be moved. the stocasticity of the leg movement was removed, ensuring that the leg that should be toggled was toggled as expected

2) MCTS
    - noisy_walker_v5 is a python translation of noisy_walker_v5a.m
    - noisy_walker_mcts is the MCTS algorithm which uses noisy_walker_v5 as a reward function by calculating the total distance traveled
    - the remaining file is a variation trying different input and output methodologies but is not used in the MCTS algorithm

3) DQN
    - noisy_walker_v5a is a variation of noisy_walker_v5 modified for providing a benchmark for noisy_walker_v6
    - noisy_walker_v6 converts the iterative method of noisy_walker_v5 to performing only one iteration. This was done in preparation for a step in the DQN algorithm
    - iterative_vs_single.py compares the results of noisy_walker_v5a and noisy_walker_v6 to ensure the outputs are the same
    - noisy_walker_dqn is a basic DQN algorithm which uses noisy_walker_v6 to calculate the next state of the cell. This currently needs modifications to enhance the model performance

4) animate_walker is a file which takes the x, y coordinates of a simulation run after either mcts or dqn so the results of the simulation can be visualized