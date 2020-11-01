## Set up environment
```bash
conda create -n matrl python=3.7
conda activate matrl
install_linux.sh # if linux
# install_mac.sh # if mac
```

### Setup mujoco
Following the instructios in https://github.com/openai/mujoco-py and https://github.com/schroederdewitt/multiagent_mujoco to setup a mujoco environment
In the end, remember to set the following environment variables:
```bash
LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco200/bin;
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

## Run experiments

### Random Matrix
```bash
python run_stateless.py
```

### Grid World
```bash
python run_experiment.py --config_file=./grid_game_configs/checker_config.yaml --seed=89757 --device=CPU
python run_experiment.py --config_file=./grid_game_configs/switch_config.yaml --seed=89757 --device=CPU
```

### Mujoco
```bash
python run_experiment.py --config_file=./mujoco_configs/hopper_matrl.yaml --seed=89757 --device=CPU
# sh run_mujoco_hopper_cpu.sh # for experiment scripts
```

### Atari
```bash
python run_experiment.py --config_file=./atari_configs/pong_matrl.yaml --seed=89757 --device=CPU
# sh run_mujoco_hopper_cpu.sh # for experiment scripts
```

## Run evaluation 

We can evalute the performance of the trained model by playing with the checkpoints to show the policy pair wise scores. The output is a matrix each cell represents the average score for playing for policy i and j. 

Evalute methods: nash_conv | population_performance

```bash
python evaluate_zero_sum_games.py --config_file=./atari_configs/pong_matrl.yaml --checkpoint_path PATH_TO_AGENT_1_CHECKPOINTS  PATH_TO_AGENT_2_CHECKPOINTS --agent_ids 0 1 --evaluate_method=population_performance 
