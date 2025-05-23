import os
import argparse
import shutil
import subprocess


def parse_case_and_curr_time(filename):
    # filename: CASE_xxx_123_127.mp4
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    parts = name.split('_')
    if len(parts) < 3:
        raise ValueError(f"Filename {filename} does not contain enough '_' for curr_time and goal_time parsing.")
    curr_time = int(parts[-2])
    goal_time = int(parts[-1])
    case = '_'.join(parts[:-2])
    return case, curr_time, goal_time

def find_init_image(raw_data_dir, case, curr_time):
    # Assume image is named as {curr_time}.jpg in raw_data_dir/case/
    img_path = os.path.join(raw_data_dir, case, f"{curr_time}.jpg")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Init image not found: {img_path}")
    return img_path

def find_pkl_file(raw_data_dir, case):
    # Find the only .pkl file in raw_data_dir/case/
    case_dir = os.path.join(raw_data_dir, case)
    pkl_files = [f for f in os.listdir(case_dir) if f.endswith('.pkl')]
    if not pkl_files:
        raise FileNotFoundError(f"No pkl file found in {case_dir}")
    if len(pkl_files) > 1:
        print(f"[WARN] Multiple pkl files found in {case_dir}, using the first one: {pkl_files[0]}")
    return os.path.join(case_dir, pkl_files[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Input folder with .mp4 files to parse')
    parser.add_argument('--raw_data_dir', type=str, required=True, help='Raw data folder with images and pkl files')
    parser.add_argument('--traj_vis_out', type=str, required=True, help='Output folder for trajectory visualization')
    parser.add_argument('--init_img_out', type=str, required=True, help='Output folder for copied init images')
    parser.add_argument('--traj_vis_script', type=str, default=os.path.join(os.path.dirname(__file__), 'traj_vis.py'), help='Path to traj_vis.py')
    args = parser.parse_args()

    os.makedirs(args.traj_vis_out, exist_ok=True)
    os.makedirs(args.init_img_out, exist_ok=True)

    for fname in os.listdir(args.input_dir):
        if not fname.endswith('.mp4'):
            continue
        try:
            case, curr_time, goal_time = parse_case_and_curr_time(fname)
        except Exception as e:
            print(f"[WARN] {e}")
            continue
        # Find pkl file
        try:
            pkl_path = find_pkl_file(args.raw_data_dir, case)
        except Exception as e:
            print(f"[WARN] {e}")
            continue
        # Call traj_vis.py
        cmd = [
            'python', args.traj_vis_script,
            '--pkl_path', pkl_path,
            '--save_dir', args.traj_vis_out,
            '--curr_time', str(curr_time),
            '--goal_time', str(goal_time)
        ]
        print(f"[INFO] Running: {' '.join(cmd)}")
        subprocess.run(cmd)
        # Copy init image
        try:
            img_path = find_init_image(args.raw_data_dir, case, curr_time)
            img_out_path = os.path.join(args.init_img_out, f"{case}_{curr_time}.jpg")
            shutil.copy(img_path, img_out_path)
        except Exception as e:
            print(f"[WARN] {e}")

if __name__ == "__main__":
    main() 