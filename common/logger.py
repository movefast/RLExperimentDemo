from collections import defaultdict

import torch as T


class MetricLogger:
    def __init__(self, exp_name, run_num, dest_dir):
        self.exp_name = exp_name
        self.run_num = run_num
        self.metrics = defaultdict(lambda: {exp_name: [[]]})
        self.dest_dir = dest_dir

    def add_scalar(self, key, value):
        self.metrics[key][self.exp_name][0].append(value)

    def add_list(self, key, value):
        self.metrics[key][self.exp_name][0].extend(value)

    def add_meta(self, key, value):
        self.metrics[key][self.exp_name] = value

    def dump(self):
        (self.dest_dir/"metrics").mkdir(parents=True, exist_ok=True)
        T.save(dict(self.metrics), self.dest_dir/f'metrics/{self.exp_name}_{self.run_num}.torch')
