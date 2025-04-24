import numpy as np
import re

def noisy_walker_v5(infile, eta, spring_constant, cirrus_force, detachment_length,\
                    stateVector, legs_to_move_vector):
    # Open the input file
    num_cirri = len(stateVector[0])
    iterations=len(stateVector)
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

    # Combine xpos and ypos into a single list of tuples for cirri positions
    cirri_pos = list(zip(xpos, ypos))

    num_cirri = current_cirrus_ID

    # print('name   xpos    ypos')
    # for i in range(num_cirri):
    #     X = f"cirrus {cirrus_name[i]} coordinates {xpos[i]}, {ypos[i]}"
    #     print(X)

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

    # Initialize variables
    directions = np.zeros(iterations)
    cellxrecord = np.zeros(iterations)
    cellyrecord = np.zeros(iterations)
    motion_state_record = np.zeros((num_cirri, iterations))
    state_sum = np.zeros(iterations)
    total_detachments = 0

    xattach = np.zeros(num_cirri)
    yattach = np.zeros(num_cirri)

    cirrus_lengths = np.zeros(num_cirri)
    motionstate = np.zeros(num_cirri, dtype=int)  # Motion state: 0 = attached, 1 = moving
    last_transition = np.zeros(num_cirri, dtype=int)

    forceX = np.zeros(num_cirri)
    forceY = np.zeros(num_cirri)

    # Set up initial conditions: cell at rest with all cirri attached and located at the origin
    cell_direction = 0  # Angle of cell long axis relative to lab frame
    cell_position_x = x_centroid  # Coordinates of cell center in lab frame
    cell_position_y = y_centroid

    for i in range(num_cirri):
        xattach[i] = xpos[i]  # Initialize attached position directly underneath
        yattach[i] = ypos[i]
        cirrus_lengths[i] = 0
        motionstate[i] = 0
        forceX[i] = 0
        forceY[i] = 0

    # Keep track of transitions and motion states
    release_spontaneous = 0
    release_pulling = 0

    total_motion = 0

    # Loop over iterations
    for current_iteration in range(iterations):
        # Calculate the sum of motion states
        sum_states = sum(motionstate)
        motion_state_record[:, current_iteration] = motionstate
        state_sum[current_iteration] = sum_states
        if sum_states == num_cirri:
            total_detachments += 1

        # Calculate new forces for each cirrus
        for i in range(num_cirri):
            if motionstate[i] == 0:
                cirrus_length = np.sqrt((xpos[i] - xattach[i]) ** 2 + (ypos[i] - yattach[i]) ** 2)
                if cirrus_length > 0:
                    stretch_force = spring_constant * cirrus_length
                    unit_x = (xattach[i] - xpos[i]) / cirrus_length
                    unit_y = (yattach[i] - ypos[i]) / cirrus_length
                    forceX[i] = stretch_force * unit_x
                    forceY[i] = stretch_force * unit_y
                else:
                    forceX[i] = 0
                    forceY[i] = 0
            else:
                forceX[i] = cirrus_force * np.cos(cell_direction)
                forceY[i] = cirrus_force * np.sin(cell_direction)
        
        # Calculate rigid-body forces and torques acting on the cell
        force_sum_x, force_sum_y, torque_sum = 0, 0, 0
        for i in range(num_cirri):
            force_sum_x += forceX[i]
            force_sum_y += forceY[i]
            position_vector_x = xpos[i] - cell_position_x
            position_vector_y = ypos[i] - cell_position_y
            cross_product = position_vector_x * forceY[i] - position_vector_y * forceX[i]
            torque_sum += cross_product

        # Apply rigid-body translations
        displacement_x = force_sum_x / eta
        displacement_y = force_sum_y / eta
        cell_position_x += displacement_x
        cell_position_y += displacement_y
        total_motion += np.sqrt(displacement_x ** 2 + displacement_y ** 2)

        for i in range(num_cirri):
            if motionstate[i] == 1:
                xattach[i] += displacement_x
                yattach[i] += displacement_y
            xpos[i] += displacement_x
            ypos[i] += displacement_y

        # Apply rigid-body rotations
        eta_rotation = 0.5 * eta * (max_distance ** 2)
        angular_displacement = torque_sum / eta_rotation
        cell_direction += angular_displacement
        cell_direction %= 2 * np.pi

        for i in range(num_cirri):
            # Apply rotation to the cirrus position
            position_x = xpos[i] - cell_position_x
            position_y = ypos[i] - cell_position_y
            new_x = position_x * np.cos(angular_displacement) - position_y * np.sin(angular_displacement)
            new_y = position_x * np.sin(angular_displacement) + position_y * np.cos(angular_displacement)
            xpos[i] = new_x + cell_position_x
            ypos[i] = new_y + cell_position_y

            # Apply rotation to substrate attachment point
            if motionstate[i] == 1:
                position_x = xattach[i] - cell_position_x
                position_y = yattach[i] - cell_position_y
                new_x = position_x * np.cos(angular_displacement) - position_y * np.sin(angular_displacement)
                new_y = position_x * np.sin(angular_displacement) + position_y * np.cos(angular_displacement)
                xattach[i] = new_x + cell_position_x
                yattach[i] = new_y + cell_position_y
        
        ### Modified to only move one leg at once
        ### The RL model will assume a leg transition is always successful
        ### For this reason, P01 and P10 have been removed
        # check for motion state transitions
        # Check for detachment of cirri due to stretch
        detached = []
        for i in range(num_cirri):
            cirrus_length = np.sqrt((xpos[i] - xattach[i]) ** 2 + (ypos[i] - yattach[i]) ** 2)
            if motionstate[i] == 0 and cirrus_length > detachment_length:
                motionstate[i] = 1
                last_transition[i] = current_iteration
                xattach[i] = xpos[i]
                yattach[i] = ypos[i]
                release_pulling += 1
                detached.append(i)
        
        if legs_to_move_vector[current_iteration] != None and legs_to_move_vector[current_iteration] not in detached:
            i=legs_to_move_vector[current_iteration]
            if motionstate[i] == 1:
                new_state = 0
                last_transition[i] = current_iteration
                xattach[i] = xpos[i]  # Attach to current position
                yattach[i] = ypos[i]
            elif motionstate[i] == 0:
                new_state = 1
                last_transition[i] = current_iteration
                xattach[i] = xpos[i]  # Detach and let substrate track cirrus base
                yattach[i] = ypos[i]
                release_spontaneous += 1
            else:
                new_state = motionstate[i]
            motionstate[i] = new_state
        
        # Save a record of the current state for plotting later
        directions[current_iteration] = cell_direction
        cellxrecord[current_iteration] = cell_position_x
        cellyrecord[current_iteration] = cell_position_y
    
    order_param = np.mean(motion_state_record[-1])
    
    times_detached = total_detachments
    summed_displacements = total_motion
    distance_travelled = np.sqrt((cell_position_x-x_centroid)**2 + (cell_position_y-y_centroid)**2)
    b=order_param
    return cellxrecord, cellyrecord, distance_travelled
