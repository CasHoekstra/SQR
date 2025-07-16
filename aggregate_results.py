import os
import json
import statistics
from collections import defaultdict

def summarize_results(directory):
    results_by_tau = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Loop through all JSON files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                for model, model_data in data.items():
                    tau = model_data.get('tau')
                    for metric in ['losses', 'coverage', 'complexity']:
                        if metric in model_data:
                            for dataset_id, values in model_data[metric].items():
                                avg_value = statistics.mean(values)
                                results_by_tau[tau][model][metric].append(avg_value)

    # Print or return summary
    for tau, models in results_by_tau.items():
        print(f"=== Tau = {tau} ===")
        for model, metrics in models.items():
            print(f"  Model: {model}")
            for metric, values in metrics.items():
                overall_avg = statistics.mean(values) if values else float('nan')
                print(f"    {metric}: avg over datasets = {overall_avg:.4f}")
        print()

# Example usage
if __name__ == "__main__":
    summarize_results("results/")