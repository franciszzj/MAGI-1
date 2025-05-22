import os
import random
import numpy as np
import argparse
import pickle

# Import torch and MagiPipeline only when needed to avoid unnecessary import errors

def generate_prompt_1(actions):
    """
    Generate a descriptive prompt using template 1.
    """
    prompt_move_list = [
        "Moves to ({:.4f}, {:.4f}) and rotates to {:.4f} degrees, viewing the environment from this perspective.",
        "Moves to ({:.4f}, {:.4f}) with orientation {:.4f} degrees, observing how the surroundings change from the previous location.",
        "Proceeds to ({:.4f}, {:.4f}) at orientation {:.4f} degrees, continuing its observation of the evolving visual scene."
    ]
    (x_0, y_0, yaw_0) = actions[0]
    assert len(actions) > 1, "actions should have at least 2 elements"
    prompt_template_1 = "Starting from position ({:.4f}, {:.4f}) with an orientation of {:.4f} degrees, the robot initially observes the scene directly ahead. It then moves sequentially through a series of positions and orientations as follows:".format(
        x_0, y_0, yaw_0)
    if len(actions) > 2:
        for (x, y, yaw) in actions[1:-1]:
            prompt_template_1 += random.choice(
                prompt_move_list).format(x, y, yaw)
    if len(actions) > 1:
        (x_N, y_N, yaw_N) = actions[-1]
        prompt_template_1 += "Finally, arrives at position ({:.4f}, {:.4f}) oriented at {:.4f} degrees, experiencing the final view of the sequence.".format(
            x_N, y_N, yaw_N)
    prompt_template_1 += "Generate a realistic video illustrating precisely what the robot would perceive visually throughout this sequence, accurately reflecting changes in its position (x, y) and orientation (yaw)."
    return prompt_template_1

def generate_prompt_2(actions):
    """
    Generate a descriptive prompt using template 2.
    """
    (x_0, y_0, yaw_0) = actions[0]
    prompt_template_2 = "The robot initially observes the scene from position ({:.4f}, {:.4f}) at orientation {:.4f} degrees. Next, the robot follows a precisely defined trajectory through exactly {} subsequent positions, listed explicitly as follows (each formatted as (x, y, yaw)):".format(
        x_0, y_0, yaw_0, len(actions) - 1)
    for (x, y, yaw) in actions[1:]:
        prompt_template_2 += "({:.4f}, {:.4f}, {:.4f}), ".format(x, y, yaw)
    # Remove the last comma and space
    prompt_template_2 = prompt_template_2[:-2]
    prompt_template_2 += ". Each tuple represents a distinct spatial coordinate (x, y) and viewing orientation yaw (in degrees). The robot moves continuously and smoothly along this exact sequence. Generate a realistic video accurately illustrating the robot's continuously evolving visual perception throughout this explicitly defined {}-position path.".format(len(actions))
    return prompt_template_2

def yaw_rotmat(yaw: float) -> np.ndarray:
    """Return a 2D or 3D rotation matrix for a given yaw angle."""
    return np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw),  np.cos(yaw), 0.0],
        [0.0,          0.0,         1.0],
    ])

def to_local_coords(positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float) -> np.ndarray:
    """
    Convert positions to local coordinates.
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError("positions must have last dim 2 or 3")
    return (positions - curr_pos).dot(rotmat)

def angle_difference(theta1, theta2):
    """
    Compute the difference between two angles, result in [-pi, pi].
    """
    delta_theta = theta2 - theta1
    delta_theta = delta_theta - 2 * np.pi * \
        np.floor((delta_theta + np.pi) / (2 * np.pi))
    return delta_theta

def compute_actions(
    traj_data,
    curr_time,
    goal_time,
    len_traj_pred,
    metric_waypoint_spacing,
    normalize
):
    """
    Compute action sequence from trajectory data.
    """
    start_index = curr_time
    end_index = curr_time + len_traj_pred + 1
    yaw = traj_data["yaw"][start_index:end_index]
    positions = traj_data["position"][start_index:end_index]

    if len(yaw.shape) == 2:
        yaw = yaw.squeeze(1)

    if yaw.shape != (len_traj_pred + 1,):
        raise ValueError("Yaw shape mismatch.")

    waypoints_pos = to_local_coords(positions, positions[0], yaw[0])
    waypoints_yaw = angle_difference(yaw[0], yaw)
    actions = np.concatenate(
        [waypoints_pos, waypoints_yaw.reshape(-1, 1)], axis=-1)
    actions = actions[1:]

    if normalize:
        actions[:, :2] /= metric_waypoint_spacing

    return actions

def pick_best_gpu():
    """
    Pick the GPU with the most free memory. Returns the GPU index as a string.
    """
    import torch
    if not torch.cuda.is_available():
        return '0'
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        max_free_mem = -1
        best_gpu = 0
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            if meminfo.free > max_free_mem:
                max_free_mem = meminfo.free
                best_gpu = i
        pynvml.nvmlShutdown()
        return str(best_gpu)
    except Exception:
        # fallback to 0 if pynvml not available or error
        return '0'

def setup_environment():
    """
    Set up environment variables for distributed and CUDA settings.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(random.randint(10000, 20000))
    os.environ['GPUS_PER_NODE'] = '1'
    os.environ['NNODES'] = '1'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = pick_best_gpu()
    
    os.environ['PAD_HQ'] = '1'
    os.environ['PAD_DURATION'] = '1'
    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['OFFLOAD_T5_CACHE'] = 'true'
    os.environ['OFFLOAD_VAE_CACHE'] = 'true'
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9;9.0'

def run_magi_inference(
    prompt,
    mode,
    output_path,
    config_path,
    image_path=None,
    prefix_video_path=None
):
    """
    Directly call MagiPipeline for inference in Python.
    """
    from inference.pipeline import MagiPipeline
    pipeline = MagiPipeline(config_path)
    if mode == 't2v':
        pipeline.run_text_to_video(prompt=prompt, output_path=output_path)
    elif mode == 'i2v':
        if image_path is None:
            raise ValueError('image_path is required for i2v mode')
        pipeline.run_image_to_video(
            prompt=prompt, image_path=image_path, output_path=output_path)
    elif mode == 'v2v':
        if prefix_video_path is None:
            raise ValueError('prefix_video_path is required for v2v mode')
        pipeline.run_video_to_video(
            prompt=prompt, prefix_video_path=prefix_video_path, output_path=output_path)
    else:
        raise ValueError(f'Unknown mode: {mode}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path', type=str,
                        help='Path to the pkl file containing trajectory data')
    parser.add_argument('--input_image', type=str, default=None,
                        help='Path to the input image (for i2v)')
    parser.add_argument('--input_video', type=str, default=None,
                        help='Path to the input video (for v2v)')
    parser.add_argument('--output_path', type=str,
                        required=True, help='Path to save the output video')
    parser.add_argument('--config_path', type=str,
                        default='example/4.5B/4.5B_base_config.json', help='Config file path')
    parser.add_argument(
        '--mode', type=str, choices=['t2v', 'i2v', 'v2v'], required=True, help='Mode: t2v, i2v, v2v')
    parser.add_argument('--curr_time', type=int, default=0,
                        help='Current time index for trajectory')
    parser.add_argument('--goal_time', type=int, default=32,
                        help='Goal time index for trajectory')
    parser.add_argument('--len_traj_pred', type=int, default=32,
                        help='Length of trajectory prediction')
    parser.add_argument('--metric_waypoint_spacing', type=float,
                        default=0.25, help='Normalization factor for positions')
    parser.add_argument('--normalize', action='store_true',
                        help='Whether to normalize output')
    parser.add_argument('--prompt_type', type=int,
                        help='Prompt type: 1 or 2. If not provided, will be determined by len_traj_pred.')
    args = parser.parse_args()

    # Read pkl and generate actions
    if args.pkl_path is not None:
        with open(args.pkl_path, 'rb') as f:
            traj_data = pickle.load(f)
        actions = compute_actions(
            traj_data,
            curr_time=args.curr_time,
            goal_time=args.goal_time,
            len_traj_pred=args.len_traj_pred,
            metric_waypoint_spacing=args.metric_waypoint_spacing,
            normalize=args.normalize
        )
        actions = actions.tolist()
        # If prompt_type is provided, use it; otherwise, use len_traj_pred to decide
        if args.prompt_type is not None:
            if args.prompt_type == 1:
                prompt = generate_prompt_1(actions)
            else:
                prompt = generate_prompt_2(actions)
        else:
            if args.len_traj_pred > 4:
                prompt = generate_prompt_2(actions)
            else:
                prompt = generate_prompt_1(actions)
    else:
        prompt = "Please provide a prompt or pkl_path."

    setup_environment()

    # Inference
    try:
        if args.mode == 't2v':
            run_magi_inference(prompt, 't2v', args.output_path, args.config_path)
        elif args.mode == 'i2v':
            run_magi_inference(prompt, 'i2v', args.output_path,
                               args.config_path, image_path=args.input_image)
        elif args.mode == 'v2v':
            run_magi_inference(prompt, 'v2v', args.output_path,
                               args.config_path, prefix_video_path=args.input_video)
    except Exception as e:
        print(f"[Error] Inference failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
