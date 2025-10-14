#!/usr/bin/env python3
"""
Script to split OfficeBench tasks into balanced train and test sets.

This script reads non_image_tasks.txt and splits the tasks by difficulty level
(1-*, 2-*, 3-*) to ensure balanced representation in both train and test sets.
"""

import random
from collections import defaultdict
from pathlib import Path
import argparse


def parse_task_id(task_id: str) -> tuple:
    """Parse task ID to extract difficulty level and task number.
    
    Args:
        task_id: Task ID in format 'level-number' (e.g., '1-5', '2-13')
        
    Returns:
        Tuple of (level, number)
    """
    parts = task_id.split('-')
    return int(parts[0]), int(parts[1])


def group_tasks_by_level(task_ids: list) -> dict:
    """Group tasks by difficulty level.
    
    Args:
        task_ids: List of task IDs
        
    Returns:
        Dictionary mapping level to list of task IDs
    """
    groups = defaultdict(list)
    for task_id in task_ids:
        level, _ = parse_task_id(task_id)
        groups[level].append(task_id)
    return dict(groups)


def split_tasks_balanced(task_ids: list, test_ratio: float = 0.5, random_seed: int = 42) -> tuple:
    """Split tasks into balanced train and test sets.
    
    Args:
        task_ids: List of all task IDs
        test_ratio: Proportion of tasks to put in test set (default: 0.5 for 50/50 split)
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_tasks, test_tasks)
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Group tasks by difficulty level
    groups = group_tasks_by_level(task_ids)
    
    train_tasks = []
    test_tasks = []
    
    print("Task distribution by difficulty level:")
    print("-" * 50)
    
    for level in sorted(groups.keys()):
        level_tasks = groups[level]
        level_tasks.sort()  # Sort for consistency
        
        # Shuffle tasks within each level
        shuffled_tasks = level_tasks.copy()
        random.shuffle(shuffled_tasks)
        
        # Calculate split point
        total_count = len(shuffled_tasks)
        test_count = int(total_count * test_ratio)
        train_count = total_count - test_count
        
        # Split the tasks
        level_test = shuffled_tasks[:test_count]
        level_train = shuffled_tasks[test_count:]
        
        train_tasks.extend(level_train)
        test_tasks.extend(level_test)
        
        print(f"Level {level}: {total_count} total → {train_count} train, {test_count} test")
        print(f"  Train: {sorted(level_train)}")
        print(f"  Test:  {sorted(level_test)}")
        print()
    
    # Sort final lists for consistency
    train_tasks.sort()
    test_tasks.sort()
    
    return train_tasks, test_tasks


def save_task_list(task_ids: list, filename: str):
    """Save task list to file.
    
    Args:
        task_ids: List of task IDs
        filename: Output filename
    """
    with open(filename, 'w') as f:
        for task_id in task_ids:
            f.write(f"{task_id}\n")
    print(f"Saved {len(task_ids)} tasks to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Split OfficeBench tasks into train and test sets")
    parser.add_argument("--input", default="non_image_tasks.txt", 
                       help="Input file containing all task IDs")
    parser.add_argument("--test-ratio", type=float, default=0.5,
                       help="Proportion of tasks for test set (default: 0.5)")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for reproducible splits (default: 42)")
    parser.add_argument("--train-output", default="train_tasks.txt",
                       help="Output file for train tasks")
    parser.add_argument("--test-output", default="test_tasks.txt", 
                       help="Output file for test tasks")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show split preview without saving files")
    
    args = parser.parse_args()
    
    # Read input task list
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found!")
        return
    
    with open(args.input, 'r') as f:
        all_tasks = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(all_tasks)} tasks from {args.input}")
    print(f"Using random seed: {args.random_seed}")
    print(f"Test ratio: {args.test_ratio:.1%}")
    print()
    
    # Split tasks
    train_tasks, test_tasks = split_tasks_balanced(
        all_tasks, 
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )
    
    print("=" * 50)
    print("FINAL SPLIT SUMMARY")
    print("=" * 50)
    print(f"Total tasks: {len(all_tasks)}")
    print(f"Train tasks: {len(train_tasks)} ({len(train_tasks)/len(all_tasks):.1%})")
    print(f"Test tasks:  {len(test_tasks)} ({len(test_tasks)/len(all_tasks):.1%})")
    print()
    
    if args.dry_run:
        print("DRY RUN - Files not saved")
        print(f"Would save train tasks to: {args.train_output}")
        print(f"Would save test tasks to: {args.test_output}")
    else:
        # Save the splits
        save_task_list(train_tasks, args.train_output)
        save_task_list(test_tasks, args.test_output)
        print("\nSplit completed successfully!")


if __name__ == "__main__":
    main()
