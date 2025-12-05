# Run SAC Flow-t
MUJOCO_GL=egl python main_action_reg_three_phase.py \
 --run_group=reproduce \
 --agent=agents/acfql_transformer_ablation_online_sac.py \
 -agent.alpha=100 \
 --env_name=square-mh-low_dim \
 --sparse=False \
 --horizon_length=5



# Run SAC Flow-g
MUJOCO_GL=egl python main_action_reg_three_phase.py \
 --run_group=reproduce \
 --agent=agents/acfql_gru_ablation_online_sac.py \
 --agent.alpha=100 \
 --env_name=square-mh-low_dim \
 --sparse=False \
 --horizon_length=5


# Run SAC FloW
MUJOCO_GL=egl python main_action_reg_three_phase.py \
 --run_group=reproduce \
 --agent=agents/acfql_ablation_online.py \
 --agent.alpha=100 \
 --env_name=square-mh-low_dim \
 --sparse=False \
 --horizon_length=5


#acfql

MUJOCO_GL=egl  python main.py \
 --run_group=reproduce \
 --agent.alpha=100 \
 --env_name=square-mh-low_dim  \
 --sparse=False \
 --horizon_length=5


 #sac
python SAC_flow_transformer_jax.py 