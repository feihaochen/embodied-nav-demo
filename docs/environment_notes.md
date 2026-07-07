# Environment Notes

This project was developed on Windows 10 + WSL2.

Current diagnosis:
- Habitat-Sim viewer works.
- Habitat-Sim non-visual simulation works.
- Habitat-Sim Python RGB-D offscreen rendering fails in the conda binary environment with WindowlessEGL context creation error.
- Next attempt: build Habitat-Sim from source with CUDA disabled to avoid CUDA/EGL device matching on WSL2.

Runtime policy:
- Agent should only use RGB, depth, and robot proprioceptive state.
- No simulator object pose, semantic oracle, or shortest-path follower is used by the Agent.
