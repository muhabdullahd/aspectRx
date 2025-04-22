import json

def main():
    # Load stored metrics from the JSON file
    metrics_file = "metrics.json"
    try:
        with open(metrics_file, "r") as f:
            metrics_list = json.load(f)
    except FileNotFoundError:
        print(f"File '{metrics_file}' not found. Make sure you have run training with the SaveMetricsCallback.")
        return

    # Print all evaluation metrics
    print("Evaluation Metrics (per evaluation step):")
    for m in metrics_list:
        epoch = m.get("epoch", "N/A")
        accuracy = m.get("eval_accuracy", None)
        f1 = m.get("eval_f1", None)
        loss = m.get("eval_loss", None)
        if accuracy is not None and f1 is not None and loss is not None:
            print(f"Epoch: {epoch}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Loss: {loss:.4f}")
        else:
            print(f"Epoch: {epoch}, Metrics not complete: {m}")

    # Find the best evaluation based on accuracy
    best = max(metrics_list, key=lambda x: x.get("eval_accuracy", 0))
    print("\nBest Evaluation:")
    print(f"Epoch: {best.get('epoch', 'N/A')}")
    print(f"Accuracy: {best.get('eval_accuracy'):.4f}")
    print(f"F1 Score: {best.get('eval_f1'):.4f}")

if __name__ == "__main__":
    main()