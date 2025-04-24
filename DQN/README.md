noisy_walker_dqn attempts to better the performance of the mcts algorithm. currently, the model performs below expectations as it currently gets caught in a loop of toggling one leg.

The reward function is the distance travelled in the x-direction from the previous cell state to the current x position. The only time this changes is if the 3 leg contraint (descibed in the MCTS README) is violated, and the reward then becomes -5.

input: initial leg location, leg state, action, cell position/orientation
output: new leg location, new leg state, new cell position/orientation, reward
reward: maximize disaplacement along current orientation

Potential Future improvements to the model:
- add 15th state: no action
- try testing the model every X episodes to plot rewards and loss functions over the episodes
- randomize initial state to increase states explored over the episodes
- try increasing num steps in episode to further exploration
- visualize visited states to get an idea on how many states are actually being explored during episodes