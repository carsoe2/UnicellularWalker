import random
import math
import noisy_walker_v5 as walker
import animate_walker as animator

# Noisy Walker Parameters
infile = 'input_files/cirrus_testfile_full_rotation.txt'
eta = 20
spring_constant = 2.0
cirrus_force = 1
detachment_length = 0.5
num_legs = 14

# Action space
actions = ['leg1', 'leg2', 'leg3', 'leg4', 'leg5', 'leg6', 'leg7',
            'leg8', 'leg9', 'leg10', 'leg11', 'leg12', 'leg13', 'leg14']
action_index = {'leg1':0, 'leg2':1, 'leg3':2, 'leg4':3, 'leg5':4, 'leg6':5, 'leg7':6,
            'leg8':7, 'leg9':8, 'leg10':9, 'leg11':10, 'leg12':11, 'leg13':12, 'leg14':13}

# Global variables for MCTS
N = dict()
Q = dict()

def has_at_least_three_legs_on_substrate(state_idx):
    state = list(map(int, f"{state_idx:0{num_legs}b}"))
    return state.count(0) >= 3

# Generate the transition model for the MDP
def transition_model():
    num_states = 2**num_legs
    T = {}
    for state_idx in range(num_states):
        # Convert the state index to binary vector representation
        state = list(map(int, f"{state_idx:0{num_legs}b}"))
        T[state_idx] = {}

        for action in actions:
            next_state = state.copy()
            if state[action_index[action]] == 1:
                next_state[action_index[action]] = 0
                next_state_idx = int("".join(map(str, next_state)), 2)
                T[state_idx][action] = [(next_state_idx, 1)]
            else:
                next_state[action_index[action]] = 1
                next_state_idx = int("".join(map(str, next_state)), 2)
                T[state_idx][action] = [(next_state_idx, 1)]
    return T

# Better version of calculating reward since distance travelled depends on previous movements
def calculate_total_reward(trajectory):
    state_vector = []
    legs_to_move_vector = []
    for state, leg_to_move in trajectory:
        state_list = list(map(int, f"{state:0{num_legs}b}"))
        state_vector.append(state_list)
        legs_to_move_vector.append(action_index[leg_to_move])
    if len(trajectory) > 0:
        cellxrecord, _, _ = walker.noisy_walker_v5(infile, eta, spring_constant, cirrus_force,
                        detachment_length, state_vector, legs_to_move_vector)
        if len(trajectory)==1:
            distance_travelled = cellxrecord[0]
        else:
            distance_travelled = cellxrecord[-1]-cellxrecord[-2]
    else:
        distance_travelled = 0
    return distance_travelled
    

# Discount factor (gamma)
gamma = 0.9

# Monte Carlo Tree Search Algorithm
def mcts(T, state, trajectory, depth=3, simulations=100, gamma=0.9, exploration_param=10):
    best_action = None
    
    # TODO: implement Monte Carlo tree search algorithm
    for _ in range(simulations):
        simulate(T, state, trajectory, depth, gamma, exploration_param)

    # Find the maximum score
    max_score = max(
        Q[(state, action)] if (state, action) in Q else 0
        for action in T[state].keys()
    )

    # Collect all actions with the maximum score
    best_actions = [
        action for action in T[state].keys()
        if (Q[(state, action)] if (state, action) in Q else 0) == max_score
    ]

    # Randomly select one of the best actions
    best_action = random.choice(best_actions)

    return best_action

def simulate(T, state, trajectory, depth, gamma, exploration_param):
    if depth <= 0:
        return calculate_total_reward(trajectory)
        # return calculate_total_reward(trajectory[-1:])
    
    # Filter cases where the number of legs attached to the substrate >= 3
    valid_actions = {action for action in T[state].keys()
                     for s_prime, _ in T[state][action] if has_at_least_three_legs_on_substrate(s_prime)}
    
    # Ensure the same leg cannot be moved for two turns in a row
    if len(trajectory)>0:
        previous_action = trajectory[-1][1]
        if previous_action in valid_actions:
            valid_actions.remove(previous_action)

    # Initialize visit count and Q-value for state-action pairs if not visited
    if not any((state, action) in N for action in valid_actions):
        for action in T[state].keys():
            N[(state, action)] = 0
            Q[(state, action)] = 0.0
        return calculate_total_reward(trajectory)
        # return calculate_total_reward(trajectory[-1:])
    
    action = explore(T, state, valid_actions, exploration_param)
    
    # Sample next state and reward using transition model
    ### Filter cases where the number of legs attached to the substrate >= 3
    filtered_states = [(s_prime, prob) for s_prime, prob in T[state][action] if has_at_least_three_legs_on_substrate(s_prime)]
    next_state = random.choices(
        [s_prime for s_prime, _ in filtered_states],
        weights=[prob for _, prob in filtered_states]
    )[0]
    # Calculate the reward with this state action pair as part of the trajectory
    # This is equivalent to the "next state" of the trajectory
    trajectory.append((state, action))
    reward = calculate_total_reward(trajectory)
    # return calculate_total_reward(trajectory[-1:])

    # Recursively simulate the outcome and compute the discounted reward
    q = reward + gamma * simulate(T, next_state, trajectory, depth - 1, gamma, exploration_param)

    N[(state, action)] += 1
    Q[(state, action)] += (q - Q[(state, action)]) / N[(state, action)]
    # Remove the temporary state action pair added to the trajectory
    trajectory.pop()

    return q

def bonus(Nsa, Ns):
    return float('inf') if Nsa==0 else math.sqrt(math.log(Ns) / Nsa)

def explore(T, state, valid_actions, exploration_param):
    total_visits = sum(N[(state, action)] for action in valid_actions)
    # Choose action that maximizes Q + exploration bonus
    # Find the maximum score and all actions achieving it
    max_score = max(
        Q[(state, action)] + exploration_param * bonus(N[(state, action)], total_visits)
        for action in valid_actions
    )

    # Collect all actions with the maximum score
    best_actions = [
        action for action in valid_actions
        if Q[(state, action)] + exploration_param * bonus(N[(state, action)], total_visits) == max_score
    ]

    # Randomly select one of the best actions
    return random.choice(best_actions)


# Simulate one step using MCTS
def simulate_mcts_step(T, current_state, trajectory, depth=3, simulations=100, gamma=0.9, exploration_param=10):
    
    best_action = mcts(T, current_state, trajectory, depth, simulations, gamma)
    
    if best_action is None:
        print("Reached a terminal state!")
        return current_state, None
    
    # Execute the best action by sampling the transition model
    next_state = random.choices([s for s, _ in T[current_state][best_action]], 
                                [p for _, p in T[current_state][best_action]])[0]
    
    return next_state, best_action

# Simulate multiple steps using mcts
def run_simulation_mcts(T, start_state, steps=10, depth=3, gamma=0.9, exploration_param=10):
    
    current_state = start_state
    trajectory = []
    
    for i in range(steps):
        # print(f"Step {i+1} of {steps}")
        if len(trajectory) <= 15:
            next_state, action = simulate_mcts_step(T, current_state, trajectory, depth, simulations=100, gamma=gamma, exploration_param=exploration_param)
        else:
            next_state, action = simulate_mcts_step(T, current_state, trajectory[-15:], depth, simulations=100, gamma=gamma, exploration_param=exploration_param)
        trajectory.append((current_state, action))
        
        if next_state is None or action is None:  # If we reached a terminal state
            break
        
        current_state = next_state
    
    return trajectory


# Test the online forward search algorithm
T = transition_model()

### Test mct search algorithm
start_state_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Example starting leg positions
# convert to a state value by first creating a binary representation and converting to a base 10 value
start_state = int("".join(map(str, start_state_list)), 2)
trajectory_mcts = run_simulation_mcts(T, start_state, steps=20, depth=4, gamma=0.9, exploration_param=10)

animation_trajectory = []
legs_to_move_vector = []
# Print the trajectory of states and actions using MCTS
for state, action in trajectory_mcts:
    # Convert each state back into a list form
    state_list = list(map(int, f"{state:0{num_legs}b}"))
    # print(f"State: {state_list}, Action: {action}")
    animation_trajectory.append(state_list)
    legs_to_move_vector.append(action_index[action])

cellxrecord, cellyrecord, _ = walker.noisy_walker_v5(infile, eta, spring_constant, cirrus_force,
                        detachment_length, animation_trajectory, legs_to_move_vector)

output_file = f"./simulation_results_mcts/simulation_results.txt"

# Write the data to the file
with open(output_file, "w") as f:
    # Write trajectory_mcts
    f.write("Trajectory MCTS:\n")
    for state, action in trajectory_mcts:
        f.write(f"State: {list(map(int, f'{state:0{num_legs}b}'))}, Action: {action}\n")
    
    f.write("\nCoordinates:\n")
    # Combine cellxrecord and cellyrecord into coordinate pairs
    for x, y in zip(cellxrecord, cellyrecord):
        f.write(f"({x:.6f}, {y:.6f})\n")

animator.animate_spider(animation_trajectory, cellxrecord, cellyrecord)