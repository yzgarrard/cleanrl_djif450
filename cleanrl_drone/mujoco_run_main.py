import time
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
 
# ── 1. Load model ─────────────────────────────────────────────────────────────
XML_PATH = "tacdrone.xml"
 
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)
 
# ── 2. Set initial conditions ─────────────────────────────────────────────────
mujoco.mj_resetData(model, data)
data.qpos[0] = 0.0   # x
data.qpos[1] = 0.0   # y
data.qpos[2] = 1.0   # z  (1 m above ground)
mujoco.mj_forward(model, data)
 
# ── 3. Simulation parameters ──────────────────────────────────────────────────
dt = model.opt.timestep   # 0.01 s (from XML)
 
# ── 4. Dynamic logging lists (grow as simulation runs) ────────────────────────
times_list = []
pos_list   = []
vel_list   = []
 
# ── 5. Run simulation — loop until viewer window is closed ────────────────────
print("Simulation running. Close the viewer window to stop and plot.\n")
 
with mujoco.viewer.launch_passive(model, data) as viewer:
 
    ## Switch to the named camera defined in the XML
    #viewer.cam.type     = mujoco.mjtCamera.mjCAMERA_FIXED
    #viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
    
    viewer.cam.distance  = 2.5
    viewer.cam.elevation = -20
    viewer.cam.azimuth   = 0
    
    sim_start = time.time()
 
    while viewer.is_running():
        # Log current state
        times_list.append(data.time)
        pos_list.append(data.qpos[:3].copy())
        vel_list.append(data.qvel[:3].copy())
 
        # Physics step
        mujoco.mj_step(model, data)

        # Sync viewer
        viewer.sync()
 
        # Throttle to real-time
        elapsed    = time.time() - sim_start
        sleep_time = data.time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
 
print(f"Simulation stopped at t = {data.time:.3f} s  ({len(times_list)} steps logged).")
 
# ── 6. Convert lists to arrays ────────────────────────────────────────────────
times = np.array(times_list)
pos   = np.array(pos_list)    # shape: (N, 3)
vel   = np.array(vel_list)    # shape: (N, 3)
 
# ── 7. Save data matrix ───────────────────────────────────────────────────────
log_matrix = np.column_stack([times, pos, vel])
np.savetxt("tacdrone_log.csv",
           log_matrix,
           delimiter=",",
           header="time,x,y,z,vx,vy,vz",
           comments="")
print("Data saved to tacdrone_log.csv")
 
# ── 8. Plot results ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle(f"TacDrone Simulation  |  Duration: {times[-1]:.2f} s", fontsize=14)
 
# --- Position vs Time ---
ax = axes[0]
ax.plot(times, pos[:, 0], label="x", linewidth=1.8)
ax.plot(times, pos[:, 1], label="y", linewidth=1.8)
ax.plot(times, pos[:, 2], label="z", linewidth=1.8, color="tab:green")
ax.set_ylabel("Position  [m]")
ax.legend(loc="upper right")
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_title("Position vs Time")
 
# --- Velocity vs Time ---
ax = axes[1]
ax.plot(times, vel[:, 0], label="vx", linewidth=1.8)
ax.plot(times, vel[:, 1], label="vy", linewidth=1.8)
ax.plot(times, vel[:, 2], label="vz", linewidth=1.8, color="tab:green")
ax.set_ylabel("Velocity  [m/s]")
ax.set_xlabel("Time  [s]")
ax.legend(loc="lower left")
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_title("Velocity vs Time")
 
plt.tight_layout()
plt.savefig("tacdrone_plots.png", dpi=150)
plt.show()
print("Plots saved to tacdrone_plots.png")
