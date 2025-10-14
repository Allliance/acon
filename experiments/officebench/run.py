import json
import os
import fire
import shutil
from datetime import datetime
import pytz
from termcolor import cprint

from productive_agents.env.officebench import OfficeBenchEnv, OfficeBenchEnvConfig
from productive_agents.agents.officebench import OfficeBenchAgent, OfficeBenchAgentConfig
from experiment_config import OfficeBenchExperimentConfig, load_experiment_config

def main(
         model_name='gpt-4o',
         task_dir='tasks/1-1',
         task_config_file='tasks/1-1/subtasks/0.json',
         output_dir='outputs',
         task=None,
         trial=None,
         tag=None,
         split='train',
         mode='default',
         exp_config: OfficeBenchExperimentConfig | str | dict = None,
         local_workdir="./local_workdir",
         debug_mode=True,
         lora_name=None,
         co_config=None,
    ):
    
    
    # Set up cache handling based on mode
    assert mode in ['force_new', 'use_llm_cache', 'default']
    force_new = mode == 'force_new'
    use_llm_cache = mode == 'use_llm_cache'

    # Set up local workdir
    if local_workdir is None:
        raise ValueError("local_workdir must be specified")
    os.makedirs(local_workdir, exist_ok=True)
    
    if force_new and os.path.exists(local_workdir):
        print(f"Force new mode: removing existing local workdir: {local_workdir}")
        shutil.rmtree(local_workdir)
    
    # Copy task files to local workdir
    shutil.copytree(task_dir, os.path.join(local_workdir, task_dir), dirs_exist_ok=True)
    task_config_file = os.path.join(local_workdir, task_config_file)
    task_dir = os.path.join(local_workdir, task_dir)

    # Load task configuration
    task_config = json.load(open(task_config_file))
    task = task_config.get('task', task)
    subtask_id = task_config_file.split('/')[-1].split('.')[0]
    
    # Set up experiment tag and output directory
    if tag is None:
        timezone = pytz.timezone('America/Los_Angeles')
        tag = datetime.now(timezone).strftime('%Y%m%d%H%M%S')
    
    if output_dir is None:
        output_dir = f'{task_dir}/outputs/{subtask_id}/{model_name.replace("/", "_")}_{tag}'
        if trial is not None:
            output_dir += f'_{trial}'

    # Print experiment metadata
    attrs = ['bold']
    def black_on_white(text):
        cprint(text, 'black', 'on_white', attrs=attrs)
    
    black_on_white('-------------------META DATA INFORMATION-------------------')
    cprint(f'TASK DIRECTORY {task_dir}', attrs=attrs)
    cprint(f'TASK: {task}', attrs=attrs)
    cprint(f'Subtask ID: {subtask_id}', attrs=attrs)
    cprint(f'Config File: {task_config_file}', attrs=attrs)
    cprint(f'Output Directory: {output_dir}', attrs=attrs)
    black_on_white('-------------------META DATA INFORMATION-------------------')
    
    # Handle LLM cache
    llm_cache = None
    if os.path.exists(output_dir) and use_llm_cache and not force_new:
        assert os.path.exists(f'{output_dir}/llm_history.json'), f"LLM history not found: {output_dir}/llm_history.json"
        llm_cache = {}
        llm_history = json.load(open(f'{output_dir}/llm_history.json'))
        for item in llm_history:
            llm_cache[item[1]] = item[2]

    # Check if output already exists
    if os.path.exists(output_dir) and not force_new and not use_llm_cache:
        print(f"Output directory already exists: {output_dir}")
        return
    
    if os.path.exists(output_dir) and force_new:
        print(f"Force new mode: removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    # Load or create experiment configuration
    if exp_config is None:
        exp_config = OfficeBenchExperimentConfig()
    elif isinstance(exp_config, dict):
        exp_config = OfficeBenchExperimentConfig.from_dict(exp_config)
    elif isinstance(exp_config, str):
        # Assume it's a file path
        exp_config = OfficeBenchExperimentConfig.from_yaml_file(exp_config)
    
    # Override config with runtime parameters
    exp_config.task = task
    exp_config.local_workdir = local_workdir
    exp_config.task_dir = task_dir
    exp_config.model_name = model_name
    exp_config.debug_mode = debug_mode
    exp_config.lora_name = lora_name
    exp_config.llm_cache = llm_cache
    if co_config is not None:
        exp_config.co_config = co_config
    
    # Create environment and agent configurations
    env_config = exp_config.to_env_config()
    agent_config = exp_config.to_agent_config()
    
    # Set up task data path and create environment
    print('Local workdir:', local_workdir)
    print('Task dir:', task_dir)
    task_config["testbed_data_path"] = f'{task_dir}/testbed/data'

    env = OfficeBenchEnv(config=env_config)
    env.reset()

    # Set up output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy testbed and reference files
    if os.path.exists(f'{task_dir}/testbed'):
        shutil.copytree(f'{task_dir}/testbed', f'{output_dir}/cache/testbed', dirs_exist_ok=True)
    else:
        os.makedirs(f'{output_dir}/cache/testbed', exist_ok=True)         
    
    if os.path.exists(f'{task_dir}/reference'):
        shutil.copytree(f'{task_dir}/reference', f'{output_dir}/reference', dirs_exist_ok=True)

    # Create agent
    api_key = ""  # API key should be set via environment variables
    agent = OfficeBenchAgent(
        model_name=model_name,
        key=api_key,
        env=env,
        task_config=task_config,
        llm_cache=llm_cache,
        debug_mode=debug_mode,
        exp_config=agent_config,
        lora_name=lora_name,
    )
    
    # Execute the agent
    print(f"Starting agent execution with max_iter={exp_config.max_iter}")
    results = agent.run(env, max_iter=exp_config.max_iter)
    
    # Print execution results
    print(f"\n=== EXECUTION RESULTS ===")
    print(f"Success: {results['success']}")
    print(f"Iterations: {results['iterations']}")
    print(f"Final reward: {results['final_reward']}")
    print(f"Termination reason: {results['termination_reason']}")
    print(f"=========================\n")
    
    cprint(f"TASK {task}-{subtask_id} COMPLETED", 'green', attrs=attrs)
    
    # Save execution results
    with open(f'{output_dir}/execution_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Copy final testbed state
    work_testbed = f'{task_dir}/testbed'
    output_testbed = f'{output_dir}/testbed'
    if os.path.exists(work_testbed):
        print(f"Copying testbed from {work_testbed} to {output_testbed}")
        shutil.copytree(work_testbed, output_testbed, dirs_exist_ok=True)
    else:
        print(f"Testbed not found in {work_testbed}, skipping copy")
    
    # Save execution history
    if hasattr(env, 'dump_history'):
        env.dump_history(output_dir)
    if hasattr(agent, 'dump_history'):
        agent.dump_history(output_dir)
    
    # Save configuration used for this run
    config_for_save = {
        'exp_id': exp_config.exp_id,
        'model_name': exp_config.model_name,
        'task': exp_config.task,
        'task_dir': exp_config.task_dir,
        'max_iter': exp_config.max_iter,
        'debug_mode': exp_config.debug_mode,
        'use_workflow_memory': exp_config.use_workflow_memory,
        'use_thinking_tokens': exp_config.use_thinking_tokens,
        'prompt_file': exp_config.prompt_file,
        'co_config': exp_config.co_config
    }
    
    with open(f'{output_dir}/settings.json', 'w') as f:
        json.dump(config_for_save, f, indent=2)
    
    env.close()

if __name__ == '__main__':
    fire.Fire(main)
