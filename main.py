import os
import time
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import ControlEnv


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

# step over the environment with visualization
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128,
    "has_renderer": True,  # Enable on-screen rendering
    "has_offscreen_renderer": True,  # Enable offscreen rendering for camera observations
    "render_camera": "frontview",  # Camera view for rendering
    "control_freq": 20,  # Control frequency
}
env = ControlEnv(**env_args)
env.seed(0)
# env.reset()
init_states = task_suite.get_task_init_states(task_id)  # for benchmarking purpose, we fix the a set of initial states
init_state_id = 0
env.set_init_state(init_states[init_state_id])

# Initialize the viewer by rendering once
env.env.render()

dummy_action = [0.] * 7
print("[info] Starting visualization. Press Ctrl+C to stop.")
try:
    for step in range(100):  # Increased steps for better visualization
        obs, reward, done, info = env.step(dummy_action)
        env.env.render()  # Explicitly render to show the window
        time.sleep(0.05)  # Slow down for better visualization

        if step % 10 == 0:
            print(f"[info] Step {step}")

        if done:
            print(f"[info] Task completed at step {step}")
            break
except KeyboardInterrupt:
    print("\n[info] Visualization interrupted by user")
finally:
    env.close()
    print("[info] Environment closed")