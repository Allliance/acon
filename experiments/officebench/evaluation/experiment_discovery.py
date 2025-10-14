import glob
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentDiscovery:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def discover(self):
        # find all directories in tasks/*/{output_dir}/*
        files = glob.glob(f"tasks/*/{self.output_dir}/*/*/llm_history.json")
        logger.info(f"Found {len(files)} files")
        experiment_names = set(map(lambda x: x.split("/")[-2], files))
        experiments = {}
        for experiment_trial in experiment_names:
            if re.match(r".*_(\d+)$", experiment_trial):
                experiment_name = re.match(r"(.*)_\d+$", experiment_trial).group(1)
            else:
                experiment_name = experiment_trial
            if experiment_name not in experiments:
                experiments[experiment_name] = []
            experiments[experiment_name].append(experiment_trial)
        logger.info(f"Experiments: {experiments}")
        return experiments
    
if __name__ == "__main__":
    output_dir = "outputs"
    experiment_discovery = ExperimentDiscovery(output_dir)
    experiments = experiment_discovery.discover()
    for experiment_name, trials in experiments.items():
        print(f"Experiment: {experiment_name}")
        for trial in trials:
            print(f"  Trial: {trial}")


