#!/usr/bin/env bash
# UNVERIFIED LOOK-AHEAD (read-only). No policy or simulator episode is run.
# Re-derives the cross-repo facts used by Person B's command-binding look-ahead:
#   1. checkpoint identity (git blob SHA) across rl_sar commits and Efferent
#   2. rl_sar_zoo Go2 MJCF vs vendored third_party/unitree_mujoco (byte diff after CRLF strip)
#   3. key command-interface lines quoted from pinned commits
# Usage: bash verify_lookahead_sources.sh   (needs gh CLI + curl; public repos only)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
F="$HERE/fetched"
RLSAR=376d42c9b128f963ab08579762d5a216a976ce39
EFF=f51e1af953afe1016dae9ceeff2b5328fa51781d
ZOO=7dd30bdc7806898950b354260655d5a7f0ce844e
RLAB_V21=8b447f2c9879e2051361064e70f999373fc59a47
HIM=ef289acaa62795009363b7b819c9186690630441
PYMPC=5831ef2647bfaa3092a4d12ad3c4ff48989c3fb6
GYMQ=097a8d649ae2232d3c0ed5eb83dc0e4f77483019
VEND="$HERE/../../../third_party/unitree_mujoco"

echo "== [1] checkpoint git blob SHAs (identical blob SHA => identical bytes)"
blob() { gh api "repos/$1/contents/$2?ref=$3" --jq '.sha + " size=" + (.size|tostring)'; }
echo "rl_sar@$RLSAR policy/go2/robot_lab/policy.pt   $(blob fan-ziqi/rl_sar policy/go2/robot_lab/policy.pt $RLSAR)"
echo "rl_sar@87be546 (v2.3, 2025-03-13) models/go2/robot_lab/policy.pt $(blob fan-ziqi/rl_sar src/rl_sar/models/go2/robot_lab/policy.pt 87be546a0cce5bb402ef3256b71ce104e4c2a9d8)"
echo "efferent@$EFF examples/go2_walk/policy.pt     $(blob Eximius-Labs/efferent examples/go2_walk/policy.pt $EFF)"
echo "rl_sar@$RLSAR policy/go2/himloco/himloco.pt   $(blob fan-ziqi/rl_sar policy/go2/himloco/himloco.pt $RLSAR)"
echo "rl_sar@f13aac0 (pre-v2.3) models/go2_isaacgym/himloco.pt $(blob fan-ziqi/rl_sar src/rl_sar/models/go2_isaacgym/himloco.pt f13aac008ae89d657a8bd3d860046979ff58ba6a)"
echo "robot_lab v2.1 commit date: $(gh api repos/fan-ziqi/robot_lab/commits/$RLAB_V21 --jq '.commit.committer.date + " " + (.commit.message|split("\n")[0])')"
echo "rl_sar v2.3 commit date:    $(gh api repos/fan-ziqi/rl_sar/commits/87be546a0cce5bb402ef3256b71ce104e4c2a9d8 --jq '.commit.committer.date + " " + (.commit.message|split("\n")[0])')"

echo "== [2] rl_sar_zoo@$ZOO go2 MJCF vs vendored unitree_mujoco@$(git -C "$VEND" rev-parse HEAD)"
for x in go2.xml scene.xml; do
  if diff <(tr -d '\r' < "$F/rl_sar_zoo_$ZOO/go2_description_mjcf_$x") <(tr -d '\r' < "$VEND/unitree_robots/go2/$x") >/dev/null; then
    echo "$x: IDENTICAL"; else echo "$x: DIFFERENT"; fi
done

echo "== [3] quoted command-interface lines at pinned commits"
q() { echo "--- $1"; grep -n -E "$2" "$1" | sed 's/^/    /'; }
q "$F/rl_sar_$RLSAR/policy_go2_robot_lab_config.yaml" "observations:|default_commands|commands_scale|observations_history:"
q "$F/rl_sar_$RLSAR/policy_go2_himloco_config.yaml" "observations:|default_commands|commands_scale|observations_history:"
q "$F/rl_sar_$RLSAR/policy_go2_base.yaml" "dt:|decimation:"
q "$F/rl_sar_$RLSAR/src_rl_sar_library_core_rl_sdk_rl_sdk.cpp" "control\.(x|y|yaw) [-+]=|commands_scale"
q "$F/rl_sar_$RLSAR/src_rl_sar_src_rl_sim_mujoco.cpp" "float (ly|lx|rx) =|control\.(x|y|yaw) = (ly|lx|rx)|obs\.commands\[[0-2]\] = this->control"
q "$F/rl_sar_$RLSAR/src_rl_sar_src_rl_real_go2.cpp" "control\.(x|y|yaw) = |cmd_vel\.(linear|angular)"
q "$F/rl_sar_$RLSAR/src_rl_sar_fsm_robot_fsm_go2.hpp" "config_name = "
q "$F/efferent_$EFF/examples_go2_walk_manifest.yaml" "rate_hz|vel_x|vel_y|yaw_rate|repo:|commit:"
q "$F/efferent_$EFF/src_efferent_types.py" "names: list\[str\] = field"
q "$F/efferent_$EFF/src_efferent_backends_mujoco_backend.py" "self.substeps ="
q "$F/robot_lab_$RLAB_V21/velocity_env_cfg.py" "lin_vel_x=|rel_heading_envs|heading_command=|heading_control_stiffness|rel_standing_envs|resampling_time_range|max_curriculum|self.decimation =|self.sim.dt ="
q "$F/robot_lab_$RLAB_V21/velocity_mdp_commands.py" "> 0.2"
q "$F/robot_lab_$RLAB_V21/velocity_mdp_curriculums.py" "delta_range =|torch.clamp\(lin_vel_(x|y)"
q "$F/robot_lab_$RLAB_V21/unitree_go2_rough_env_cfg.py" "actions.joint_pos.scale|base_velocity.ranges"
q "$F/isaaclab_v2.0.2/velocity_command.py" "in the base frame|heading_control_stiffness \* heading_error"
q "$F/himloco_$HIM/legged_gym_legged_gym_envs_base_legged_robot.py" "self.commands\[:, :3\] \* self.commands_scale,$|self.commands_scale = torch.tensor|0.5\*wrap_to_pi|> 0.2\)"
q "$F/himloco_$HIM/legged_gym_legged_gym_envs_base_legged_robot_config.py" "lin_vel_x = |lin_vel_y = |ang_vel_yaw = |max_curriculum|decimation = |dt =  "
echo "--- HIMLoco envs/ dirs: $(ls "$F/himloco_$HIM" >/dev/null; gh api "repos/OpenRobotLab/HIMLoco/git/trees/$HIM?recursive=1" --jq '[.tree[] | select(.path|test("envs/[a-z0-9]+/[a-z0-9_]+_config.py$")) | .path] | join(" ")')"
echo "--- HIMLoco LICENSE line 1: $(head -1 "$F/himloco_$HIM/LICENSE")"
q "$F/pympc_$PYMPC/simulation_simulation.py" "ref_base_lin_vel=|ref_base_ang_vel=|base_vel_command_type="
q "$F/pympc_$PYMPC/quadruped_pympc_config.py" "'frequency':|'horizon':|^    'dt':|'gait':|'ref_z':|'use_velocity_modulator':"
q "$F/gym_quadruped_$GYMQ/quadruped_env.py" "def target_base_vel|ref_base_ang_vel = np.array\(\[0.0, 0.0|np.clip\(self._ref_base_ang_yaw_dot|6 \* self.robot_cfg.hip_height"
q "$F/gym_quadruped_$GYMQ/robot_cfgs.py" "go2/go2.xml"
q "$F/gym_quadruped_$GYMQ/go2.xml" "<body name=\"base\"|<option"
