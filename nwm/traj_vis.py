import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

def compute_actions(traj_data, metric_waypoint_spacing=0.25, normalize=True):
    """
    Return the full actions (x, y, yaw) from traj_data.
    traj_data: dict with keys 'yaw' and 'position', each a numpy array.
    """
    positions = traj_data["position"]
    yaw = traj_data["yaw"]
    if len(positions.shape) == 3:
        positions = positions.squeeze(1)
    if len(yaw.shape) == 2:
        yaw = yaw.squeeze(1)
    actions = np.concatenate([positions, yaw.reshape(-1, 1)], axis=-1)
    if normalize:
        actions[:, :2] /= metric_waypoint_spacing
    return actions

def plot_traj(actions, save_dir, prefix, curr_time=0, goal_time=None):
    """
    Visualize the full trajectory, highlight the segment from curr_time to goal_time,
    and show yaw as arrows at each step. Save as a single jpg.
    """
    os.makedirs(save_dir, exist_ok=True)
    xs = [a[0] for a in actions]
    ys = [a[1] for a in actions]
    yaws = [a[2] for a in actions]
    N = len(actions)
    plt.figure(figsize=(7, 7))
    # Plot full trajectory
    plt.plot(xs, ys, 'bo-', label='Full Trajectory', alpha=0.5)
    # Draw yaw arrows for all points (light gray)
    for i in range(N):
        dx = 0.15 * np.cos(yaws[i])
        dy = 0.15 * np.sin(yaws[i])
        plt.arrow(xs[i], ys[i], dx, dy, head_width=0.04, head_length=0.06, fc='gray', ec='gray', alpha=0.5)
    # Highlight curr_time to goal_time
    if goal_time is not None and goal_time > curr_time and goal_time <= N:
        plt.plot(xs[curr_time:goal_time+1], ys[curr_time:goal_time+1], 'r-', linewidth=3, label='Highlighted Segment')
        plt.scatter(xs[curr_time:goal_time+1], ys[curr_time:goal_time+1], c='r', s=60)
        # Draw yaw arrows for highlighted segment (red)
        for i in range(curr_time, goal_time+1):
            dx = 0.18 * np.cos(yaws[i])
            dy = 0.18 * np.sin(yaws[i])
            plt.arrow(xs[i], ys[i], dx, dy, head_width=0.05, head_length=0.08, fc='red', ec='red', alpha=0.9)
    # Draw heading at start, curr_time, and goal_time
    for idx, color, label in zip([0, curr_time, goal_time if goal_time is not None and goal_time < N else None], ['g', 'm', 'c'], ['Start', 'Curr', 'Goal']):
        if idx is not None and 0 <= idx < N:
            dx = 0.2 * np.cos(yaws[idx])
            dy = 0.2 * np.sin(yaws[idx])
            plt.arrow(xs[idx], ys[idx], dx, dy, head_width=0.06, head_length=0.09, fc=color, ec=color, label=label)
            plt.scatter(xs[idx], ys[idx], c=color, s=100, marker='*')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Trajectory ({prefix})')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_traj_vis.jpg"))
    plt.close()

def process_pkl(pkl_path, save_dir, curr_time, goal_time, metric_waypoint_spacing, normalize):
    case_name = os.path.basename(os.path.dirname(pkl_path))
    prefix = f"{case_name}_{curr_time}_{goal_time}"
    with open(pkl_path, 'rb') as f:
        traj_data = pickle.load(f)
    actions = compute_actions(
        traj_data,
        metric_waypoint_spacing=metric_waypoint_spacing,
        normalize=normalize
    )
    plot_traj(actions, save_dir, prefix, curr_time=curr_time, goal_time=goal_time)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path', type=str, required=True, help='Path to the pkl file containing trajectory data')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save jpg images')
    parser.add_argument('--curr_time', type=int, default=0, help='Current time index for trajectory')
    parser.add_argument('--goal_time', type=int, default=4, help='Goal time index for trajectory')
    parser.add_argument('--metric_waypoint_spacing', type=float, default=0.25, help='Normalization factor for positions')
    parser.add_argument('--normalize', type=bool, default=True, help='Whether to normalize output')
    args = parser.parse_args()

    case_name = os.path.basename(os.path.dirname(args.pkl_path))
    # out_dir = os.path.join(args.save_dir, case_name)
    out_dir = args.save_dir
    process_pkl(
        args.pkl_path,
        out_dir,
        args.curr_time,
        args.goal_time,
        args.metric_waypoint_spacing,
        args.normalize
    )

if __name__ == "__main__":
    main() 