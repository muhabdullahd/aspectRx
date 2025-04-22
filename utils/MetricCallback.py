import json
from transformers import TrainerCallback

class MetricsCallback(TrainerCallback):
    def __init__(self, output_file):
        self.output_file = output_file
        self.metrics_list = []

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Select the metrics we care about. We also store the current epoch.
        selected_metrics = {
            "epoch": metrics.get("epoch", state.epoch),
            "eval_accuracy": metrics.get("eval_accuracy"),
            "eval_f1": metrics.get("eval_f1"),
            "eval_loss": metrics.get("eval_loss")
        }
        self.metrics_list.append(selected_metrics)
        # Write the list of metrics to the file
        with open(self.output_file, "w") as f:
            json.dump(self.metrics_list, f, indent=4)
        return control
