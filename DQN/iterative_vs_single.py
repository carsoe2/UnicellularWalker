import numpy as np
from noisy_walker_v5a import noisy_walker_v5
# from noisy_walker_v5b import noisy_walker_v5b
from DQN.noisy_walker_v6 import noisy_walker_v6
import re

def run_iterative_version(infile, eta, spring_constant, cirrus_force, detachment_length, stateVector, legs_to_move_vector):
    return noisy_walker_v5(infile, eta, spring_constant, cirrus_force, detachment_length, stateVector, legs_to_move_vector)

def run_single_step_version(infile, eta, spring_constant, cirrus_force, detachment_length, stateVector, legs_to_move_vector):
    num_iterations = len(stateVector)
    with open(infile, 'r') as fid:
        numlines = 0
        big_string = ""

        # Read the whole file as one huge string
        for tline in fid:
            tline = tline.strip()  # Remove newline characters and extra spaces
            if not tline.startswith('%'):  # Skip any comment lines
                numlines += 1
                big_string += ' ' + tline
    # Pad any non-space delimiters to ensure proper splitting
    big_string = re.sub(r'{', ' { ', big_string)
    big_string = re.sub(r'}', ' } ', big_string)
    big_string = re.sub(r'\[', ' [ ', big_string)
    big_string = re.sub(r'\]', ' ] ', big_string)
    big_string = re.sub(r',', ' , ', big_string)

    # Trim out multiple white spaces to prevent empty tokens
    big_string = re.sub(r'\s+', ' ', big_string).strip()

    # Split the string into tokens based on spaces
    input_words = big_string.split(' ')

    # Count the number of words
    num_words = len(input_words)

    current_cirrus_ID = 0
    cirrus_name = []
    xpos = []
    ypos = []
    leg_attach = []

    for i in range(num_words):
        word = input_words[i]
        
        if word.startswith('{'):
            current_cirrus_ID += 1  # Create a new ID for each new cirrus when defined
            cirrus_name.append(input_words[i + 1])
        elif word.startswith('['):
            number_string1 = input_words[i + 1]  # Extract the next word
            number_string2 = input_words[i + 3]  # Extract the word after the comma
            
            xpos.append(float(number_string1))  # Convert to float
            ypos.append(float(number_string2))  # Convert to float
            leg_attach.append((float(number_string1), float(number_string2)))  # Convert to float

    num_cirri = current_cirrus_ID

    # Calculate the longest linear dimension of the cell based on the largest
    # distance between pairs of cirri
    max_distance = 0
    for i in range(num_cirri):
        for j in range(num_cirri):
            distance = np.sqrt((xpos[i] - xpos[j]) ** 2 + (ypos[i] - ypos[j]) ** 2)
            if distance > max_distance:
                max_distance = distance

    # Calculate the centroid of the cell
    x_centroid = sum(xpos) / num_cirri
    y_centroid = sum(ypos) / num_cirri
    cell_state = (x_centroid, y_centroid, 0)
    leg_state = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    cellxrecord = []
    cellyrecord = []
    total_distance_traveled = []
    
    for i in range(len(legs_to_move)):
        leg_attach, leg_state, cell_state = noisy_walker_v6(leg_attach, leg_state, legs_to_move[i], cell_state)
        cellxrecord.append(cell_state[0])
        cellyrecord.append(cell_state[1])
        total_distance_traveled.append(np.sqrt((cell_state[0]-x_centroid)**2+(cell_state[1]-y_centroid)**2))
    
    return cellxrecord, cellyrecord, total_distance_traveled

def compare_versions(infile, eta, spring_constant, cirrus_force, detachment_length, stateVector, legs_to_move_vector):
    iter_x, iter_y, iter_distance = run_iterative_version(infile, eta, spring_constant, cirrus_force, detachment_length, stateVector, legs_to_move_vector)
    step_x, step_y, step_distance = run_single_step_version(infile, eta, spring_constant, cirrus_force, detachment_length, stateVector, legs_to_move_vector)
    step_x = np.array(step_x).tolist()
    step_y = np.array(step_y).tolist()
    step_distance = np.array(step_distance).tolist()
    
    # print("Final distance comparison:")
    # print(f"Iterative Version: {iter_distance}")
    # print(f"Single Step Version: {step_distance}")
    # print(f"Difference: {abs(iter_distance - step_distance)}")
    
    print("Position comparison:")
    print(f"X Difference: {np.linalg.norm(iter_x - step_x)}")
    print(f"Y Difference: {np.linalg.norm(iter_y - step_y)}")

    print("v5_x:", iter_x)
    print("v6_x:", step_x)
    print("v5_y:", iter_y)
    print("v6_y:", step_y)
    
    return np.allclose(iter_x, step_x) and np.allclose(iter_y, step_y)

# Example usage (replace with actual input values):
infile = "./input_files/cirrus_testfile_full_rotation.txt"
eta = 20
spring_constant = 2.0
cirrus_force = 1
detachment_length = 0.5
state_vector = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0],
    [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
]

legs_to_move = [10, 2, 5, 3, 2, 8, 10, 1, 8, 9, 13, 4, 4, 2, 12, 9, 4, 1, 6, 1]
is_matching = compare_versions(infile, eta, spring_constant, cirrus_force, detachment_length, state_vector, legs_to_move)
print("Versions match?", is_matching)
