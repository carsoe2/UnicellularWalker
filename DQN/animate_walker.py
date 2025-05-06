import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Ellipse

# Function to draw the spider
def draw_spider(ax, position, num_legs, leg_states):
    """Draws a spider with a specified number of legs and states."""
    body_width = 1  # Width of the spider's body
    body_height = 0.5  # Height of the spider's body
    leg_length = 1   # Total length of each leg (split into two segments)
    base_angles = np.linspace(0, 2 * np.pi, num_legs, endpoint=False)  # Leg angles
    joint_offset = 0.2  # Offset for the "knee" joint of each leg

    # Draw the body (ellipse)
    body = Ellipse(position, body_width, body_height, color='black', zorder=10)
    ax.add_patch(body)

    # Draw the legs
    for i, angle in enumerate(base_angles):
        offset = 0.1 if leg_states[i] == 1 else -0.1  # Forward or backward state
        adjusted_angle = angle + offset

        # Calculate joint (knee) position
        joint_x = position[0] + (leg_length / 2) * np.cos(adjusted_angle)
        joint_y = position[1] + (leg_length / 2) * np.sin(adjusted_angle)

        # Calculate foot position, perpendicular to the joint
        if adjusted_angle < np.pi:
            foot_angle = adjusted_angle + np.pi / 4
        else:
            foot_angle = adjusted_angle - np.pi / 4
        foot_x = joint_x + (leg_length / 2) * np.cos(foot_angle)
        foot_y = joint_y + (leg_length / 2) * np.sin(foot_angle)

        # Draw upper leg
        ax.plot([position[0], joint_x], [position[1], joint_y], color='brown', lw=3)
        # Draw lower leg
        ax.plot([joint_x, foot_x], [joint_y, foot_y], color='brown', lw=3)

# Generate an animation based on the inputs
def animate_spider(leg_matrix, x_coords, y_coords):
    """Creates an animation of the spider based on the provided inputs."""
    num_legs = len(leg_matrix[0])  # Number of legs
    num_frames = len(x_coords)    # Number of animation frames

    # Calculate limits for the animation
    margin = 2.0  # Add some margin around the trajectory
    x_min, x_max = min(x_coords) - margin, max(x_coords) + margin
    y_min, y_max = min(y_coords) - margin, max(y_coords) + margin

    fig, ax = plt.subplots()
    trail_x, trail_y = [], []  # To store the trail coordinates

    def update(frame):
        position = (x_coords[frame], y_coords[frame])
        leg_states = leg_matrix[frame]

        # Add the current position to the trail
        trail_x.append(position[0])
        trail_y.append(position[1])

        # Clear previous frame
        ax.clear()

        # Draw the trail
        ax.plot(trail_x, trail_y, color='blue', lw=2, zorder=5)

        # Draw the spider
        draw_spider(ax, position, num_legs, leg_states)

        # Set plot limits and maintain the aspect ratio
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='datalim')
        ax.axis('off')  # Turn off the axes for better visualization

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100, repeat=False)
    plt.show()