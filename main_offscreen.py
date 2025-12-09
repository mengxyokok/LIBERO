import os
import time
import numpy as np
import imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_10"  # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()

# retrieve a specific task
task_id = 0
task = task_suite.get_task(task_id)
task_name = task.name
task_description = task.language
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
      f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

# Use offscreen rendering (more reliable)
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 256,
    "camera_widths": 256,
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
env.reset()
init_states = task_suite.get_task_init_states(task_id)
init_state_id = 0
env.set_init_state(init_states[init_state_id])

dummy_action = [0.] * 7
print("[info] Starting rendering. Collecting frames for video...")
os.makedirs("renders", exist_ok=True)

frames = []

try:
    for step in range(100):
        obs, reward, done, info = env.step(dummy_action)

        # Get rendered image from observations
        if "agentview_image" in obs:
            img = obs["agentview_image"]
            # imageio uses RGB format, so keep the image as is
            frames.append(img)

            if step % 10 == 0:
                print(f"[info] Step {step}, collected frame")

        if done:
            print(f"[info] Task completed at step {step}")
            break

except KeyboardInterrupt:
    print("\n[info] Rendering interrupted by user")
finally:
    env.close()
    print("[info] Environment closed")
    
    # Save video
    if frames:
        print(f"[info] Saving video with {len(frames)} frames...")
        fps = 20
        video_path = "renders/output.mp4"
        imageio.mimwrite(video_path, frames, fps=fps, codec='libx264', quality=8)
        print(f"[info] Video saved to {video_path}")
    else:
        print("[info] No frames collected")
