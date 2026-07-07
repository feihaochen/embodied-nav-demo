from pathlib import Path
import habitat_sim

scene_path = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"

if not Path(scene_path).exists():
    raise FileNotFoundError(scene_path)

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = scene_path
sim_cfg.enable_physics = False
sim_cfg.create_renderer = False

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = []
agent_cfg.action_space = {
    "move_forward": habitat_sim.agent.ActionSpec(
        "move_forward",
        habitat_sim.agent.ActuationSpec(amount=0.25),
    ),
    "turn_left": habitat_sim.agent.ActionSpec(
        "turn_left",
        habitat_sim.agent.ActuationSpec(amount=15.0),
    ),
}

sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

try:
    agent = sim.initialize_agent(0)
    print("[OK] Simulator created with renderer disabled.")
    print("[INFO] Pathfinder loaded:", sim.pathfinder.is_loaded)
    print("[INFO] Agent position:", agent.get_state().position)

    for i in range(5):
        sim.step("turn_left")
        print(f"[OK] step {i}")

finally:
    sim.close()
