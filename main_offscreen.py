import os
import time
import numpy as np
from PIL import Image
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
print("[info] Starting rendering. Saving images to ./renders/")
os.makedirs("renders", exist_ok=True)

try:
    for step in range(100):
        obs, reward, done, info = env.step(dummy_action)

        # Get rendered image from observations
        if "agentview_image" in obs:
            img = obs["agentview_image"]
            # Convert to PIL Image and save
            img_pil = Image.fromarray(img)
            img_pil.save(f"renders/step_{step:04d}.png")

            if step % 10 == 0:
                print(f"[info] Step {step}, saved image")

        if done:
            print(f"[info] Task completed at step {step}")
            break

except KeyboardInterrupt:
    print("\n[info] Rendering interrupted by user")
finally:
    env.close()
    print("[info] Environment closed")
    print("[info] Images saved to ./renders/ directory")
    print("[info] You can create a video with: ffmpeg -framerate 20 -i renders/step_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4")
