import os
import sys
import time
import glob
import subprocess
from multiprocessing import Process, Queue, Manager
import pynvml
from tqdm import tqdm

len_traj_pred = int(os.environ.get("LEN_TRAJ_PRED", 4))
INPUT_DIR = "/media/cheliu/world_model/nwm_work_dirs/data/recon"
OUTPUT_DIR = f"/media/cheliu/world_model/nwm_work_dirs/data/recon_magi_output_len{len_traj_pred}"
PREDICT_SCRIPT = os.path.join(os.path.dirname(__file__), "predict.py")
MIN_FREE_MEM = 20 * 1024 ** 3  # 20GB
CHECK_INTERVAL = 10  # seconds
GPU_LAUNCH_INTERVAL = 120  # 2 minutes in seconds


def get_all_cases(input_dir):
    """Get all second-level folders (test cases) under input_dir."""
    return [os.path.join(input_dir, d) for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))]

def get_gpu_free_mem():
    """Return a list of (gpu_id, free_mem_bytes) for all GPUs."""
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    free_mem = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mem.append((i, meminfo.free))
    pynvml.nvmlShutdown()
    return free_mem

def find_case_files(case_dir):
    """
    Find all jpg images (sorted by numeric filename) and trajectory file in a case directory.
    Returns: (list_of_images, traj_path)
    """
    image_files = glob.glob(os.path.join(case_dir, "*.jpg"))
    # sort by numeric filename (without extension)
    try:
        image_files = sorted(image_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    except Exception:
        image_files = sorted(image_files)
    traj = glob.glob(os.path.join(case_dir, "*.pkl"))
    if not image_files or not traj:
        raise FileNotFoundError(f"Missing jpg image(s) or trajectory in {case_dir}")
    return image_files, traj[0]

def worker(case_dir, gpu_id, progress):
    case_name = os.path.basename(case_dir)
    image_files, traj_path = find_case_files(case_dir)
    for image_path in image_files:
        img_base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{case_name}_{img_base}_{int(img_base) + len_traj_pred}.mp4")
        cmd = [
            sys.executable, PREDICT_SCRIPT,
            "--mode", "i2v",
            "--input_image", image_path,
            "--pkl_path", traj_path,
            "--output_path", out_path,
            "--curr_time", str(int(img_base)),
            "--goal_time", str(int(img_base) + len_traj_pred),
            "--len_traj_pred", str(len_traj_pred),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[GPU {gpu_id}] Running: {case_name} {img_base}")
        subprocess.run(cmd, env=env)
        print(f"[GPU {gpu_id}] Finished: {case_name} {img_base}")
        progress['done'] += 1

def gpu_scheduler(cases):
    """
    Schedule all images in all cases to GPUs with >20GB free memory and show progress bar.
    Allow multiple predict.py processes on the same GPU as long as free memory > 20GB.
    Enforce a 2-minute interval between launches on the same GPU.
    """
    # Flatten all tasks (case_dir, image_path) for progress bar
    all_images = []
    case_to_images = {}
    for case in cases:
        imgs, _ = find_case_files(case)
        case_to_images[case] = imgs
        for img in imgs:
            all_images.append((case, img))
    total = len(all_images)
    manager = Manager()
    progress = manager.dict()
    progress['done'] = 0
    # For scheduling, we still use case granularity, but each worker will process all images in its case
    task_queue = Queue()
    for case in cases:
        task_queue.put(case)
    running = []  # list of (Process, gpu_id)
    last_launch_time = {}  # gpu_id -> last launch timestamp
    with tqdm(total=total, desc="Progress", ncols=80) as pbar:
        last_done = 0
        while not task_queue.empty() or running:
            # Clean up finished processes
            running = [(p, gid) for (p, gid) in running if p.is_alive()]
            # Check available GPUs (allow multiple per GPU if enough free mem)
            free_mem = get_gpu_free_mem()
            now = time.time()
            for gpu_id, mem in free_mem:
                # Enforce 2min interval between launches on the same GPU
                if gpu_id in last_launch_time and now - last_launch_time[gpu_id] < GPU_LAUNCH_INTERVAL:
                    continue
                while mem > MIN_FREE_MEM and not task_queue.empty():
                    case_dir = task_queue.get()
                    p = Process(target=worker, args=(case_dir, gpu_id, progress))
                    p.start()
                    running.append((p, gpu_id))
                    last_launch_time[gpu_id] = time.time()
                    mem -= MIN_FREE_MEM
                    # After launching one, break to enforce interval for next launch on this GPU
                    break
            # Update progress bar
            if progress['done'] > last_done:
                pbar.update(progress['done'] - last_done)
                last_done = progress['done']
            time.sleep(CHECK_INTERVAL)
        # Final update
        while last_done < total:
            if progress['done'] > last_done:
                pbar.update(progress['done'] - last_done)
                last_done = progress['done']
            time.sleep(1)
    # Wait for all to finish
    for p, _ in running:
        p.join()

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    cases = get_all_cases(INPUT_DIR)
    print(f"Found {len(cases)} cases.")
    gpu_scheduler(cases)

if __name__ == "__main__":
    main()
