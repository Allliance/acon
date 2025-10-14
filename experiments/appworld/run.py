#!/usr/bin/env python3
"""
Main entry point for running individual AppWorld tasks.

This script handles the execution of a single AppWorld task with the specified configuration.
"""

import json
import os
import shutil
from datetime import datetime
import logging
from typing import Dict, Any, Optional

from productive_agents.env.appworld import AppWorldEnv, AppWorldEnvConfig
from productive_agents.agents.appworld import AppWorldAgent, AppWorldAgentConfig


def setup_logging(debug_mode: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_experiment_config(exp_config: Dict[str, Any]) -> AppWorldAgentConfig:
    """Load experiment configuration."""
    # Get valid field names from AppWorldAgentConfig
    from productive_agents.agents.appworld.config import AppWorldAgentConfig
    valid_fields = set(AppWorldAgentConfig.__dataclass_fields__.keys())
    
    # Extract only valid parameters
    config_params = {}
    extra_config = {}
    
    for key, value in exp_config.items():
        if key in valid_fields and key != 'extra_config':
            config_params[key] = value
        else:
            extra_config[key] = value
    
    # Add extra_config if there are any extra parameters
    if extra_config:
        config_params['extra_config'] = extra_config
    
    # Remove None values to let defaults take effect
    config_params = {k: v for k, v in config_params.items() if v is not None}
    
    return AppWorldAgentConfig(**config_params)


def main(
    task_id: str,
    split: str = 'train',
    output_dir: str = 'outputs',
    exp_config: Dict[str, Any] = None,
    model_name: str = 'gpt-4o',
    debug_mode: bool = True,
    experiment_name: str = 'minimal_test',
    max_iter: int = 50,
    model_ctxopt: Optional[Any] = None,
    lora_name: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function to run a single AppWorld task.
    
    Args:
        task_id: AppWorld task ID to run
        split: Dataset split the task belongs to
        output_dir: Directory to save outputs
        exp_config: Experiment configuration dictionary
        model_name: Name of the model to use
        debug_mode: Enable debug mode
        experiment_name: Name for the AppWorld experiment
        max_iter: Maximum number of iterations
        **kwargs: Additional arguments
        
    Returns:
        Dictionary containing execution results
    """
    
    setup_logging(debug_mode)
    logger = logging.getLogger(__name__)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Default configuration
    if exp_config is None:
        exp_config = {
            'debug_mode': debug_mode,
            'max_iter': max_iter,
            'experiment_name': experiment_name
        }
    
    # Load experiment configuration
    agent_config = load_experiment_config(exp_config)
    
    # Setup environment configuration
    env_config = AppWorldEnvConfig(
        experiment_name=experiment_name,
        max_interactions=max_iter,
        debug_mode=debug_mode
    )
    
    # Print task information
    logger.info(f"Starting AppWorld task: {task_id}")
    logger.info(f"Split: {split}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Max iterations: {max_iter}")
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize environment
    env = AppWorldEnv(config=env_config)
    
    # Reset environment with the specific task
    obs = env.reset(task_id=task_id)
    logger.info(f"Environment initialized with task: {task_id}")
    
    # Get API key from environment
    api_key = ''
    
    # Create task configuration
    task_config = {
        'task_id': task_id,
        'split': split,
        'experiment_name': experiment_name,
        'username': 'user',
        'date': datetime.now().strftime('%Y-%m-%d'),
        'weekday': datetime.now().strftime('%A'),
        'time': datetime.now().strftime('%H:%M:%S'),
    }
    
    # Initialize agent
    agent = AppWorldAgent(
        model_name=model_name,
        key=api_key,
        env=env,
        task_config=task_config,
        exp_config=agent_config,
        debug_mode=debug_mode,
        model_ctxopt=model_ctxopt,
        lora_name=lora_name
    )
    
    logger.info("Agent initialized, starting execution...")
    
    # Run the agent
    results = agent.run(env, max_iter=max_iter)
    
    # Get token usage and cost information
    token_summary = agent.get_token_usage_summary()
    
    # Add additional information to results
    results.update({
        'task_id': task_id,
        'split': split,
        'model_name': model_name,
        'experiment_name': experiment_name,
        'config': exp_config,
        'token_usage': token_summary  # Add token usage info to results
    })
    
    # Save results
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save agent history
    agent.dump_history(output_dir)
    
    # Save environment trajectory
    env.dump_history(output_dir)
    
    # Get and print token usage and cost information
    token_summary = agent.get_token_usage_summary()
    if token_summary:
        print("\n" + "="*50)
        print("TOKEN USAGE AND COST SUMMARY")
        print("="*50)
        print(f"Model: {token_summary.get('model_name', 'Unknown')}")
        print(f"Total Requests: {token_summary.get('total_requests', 0)}")
        print(f"Input Tokens: {token_summary.get('total_input_tokens', 0):,}")
        print(f"Output Tokens: {token_summary.get('total_output_tokens', 0):,}")
        print(f"Total Tokens: {token_summary.get('total_tokens', 0):,}")
        print("-" * 30)
        print(f"Input Cost: ${token_summary.get('input_cost_usd', 0):.6f}")
        print(f"Output Cost: ${token_summary.get('output_cost_usd', 0):.6f}")
        print(f"TOTAL COST: ${token_summary.get('total_cost_usd', 0):.6f}")
        print("="*50)
    
    # Print results summary
    logger.info(f"Task completed!")
    logger.info(f"Success: {results['success']}")
    logger.info(f"Iterations: {results['iterations']}")
    logger.info(f"Termination reason: {results['termination_reason']}")
    logger.info(f"Final reward: {results['final_reward']}")
    
    # Clean up
    env.close()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a single AppWorld task")
    parser.add_argument("--task_id", type=str, required=True, help="Task ID to run")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name")
    parser.add_argument("--debug_mode", action="store_true", help="Enable debug mode")
    parser.add_argument("--max_iter", type=int, default=50, help="Maximum iterations")
    parser.add_argument("--experiment_name", type=str, default="minimal_test", help="Experiment name")
    
    args = parser.parse_args()
    
    result = main(
        task_id=args.task_id,
        split=args.split,
        output_dir=args.output_dir,
        model_name=args.model_name,
        debug_mode=args.debug_mode,
        max_iter=args.max_iter,
        experiment_name=args.experiment_name
    )
    
    print(f"Task {args.task_id} completed with result: {result}")
