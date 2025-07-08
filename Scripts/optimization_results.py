import json
import pandas as pd
import sys

def json_to_csv(json_path, output_csv="optimization_cluster_final_fine_grained.csv"):
    with open(json_path, "r") as f:
        data = json.load(f)

    configs = data["configs"]
    runs = data["data"]
    config_origins = data["config_origins"]

    records = []

    for run in runs:
        config_id = str(run["config_id"])
        cost = run.get("cost", None)
        budget = run.get("budget", None)
        error = run.get("additional_info", {}).get("error", "")
        acquisition = config_origins.get(config_id, "")

        config = configs.get(config_id, None)
        if config:
            record = {
                "config_id": config_id,

                "hidden_fc1": config["hidden_fc1"],
                "instrument_emb": config["instrument_embedding_dim"],
                "learning_rate": config["learning_rate"],

                "acquisition": acquisition,
                "budget": budget,
                "cost": cost,
                "error": error
            }
            # record = {
            #     "config_id": config_id,
            #     "d_model": config["d_model"],
            #     "dropout": config["dropout"],
            #     "hidden_fc1": config["hidden_fc1"],
            #     "instrument_emb": config["instrument_embedding_dim"],
            #     "learning_rate": config["learning_rate"],
            #     "n_layers": config["n_layers"],
            #     "optimizer_name": config["optimizer_name"],
            #     "acquisition": acquisition,
            #     "budget": budget,
            #     "cost": cost,
            #     "error": error
            # }
            records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved to {output_csv}")

# Example usage:
json_to_csv("/Users/madinabekbergenova/PycharmProjects/pythonProject/optimization_cluster_12_05/optimization_cluster_fine_grained_27_06/0/runhistory.json")

import json
import pandas as pd

def append_json_to_csv_with_gap(json_path, output_csv="/Users/madinabekbergenova/PycharmProjects/pythonProject/Optimization_results_15_04/optimization_output_devices.csv"):
    # Load the existing CSV file (if it exists)
    try:
        df_existing = pd.read_csv(output_csv)
    except FileNotFoundError:
        df_existing = pd.DataFrame()

    # Load new JSON data
    with open(json_path, "r") as f:
        data = json.load(f)

    configs = data["configs"]
    runs = data["data"]
    config_origins = data["config_origins"]

    new_records = []

    for run in runs:
        config_id = str(run["config_id"])
        cost = run.get("cost", None)
        budget = run.get("budget", None)
        error = run.get("additional_info", {}).get("error", "")
        acquisition = config_origins.get(config_id, "")

        config = configs.get(config_id, None)
        if config:
            record = {
                "config_id": config_id,
                "batch_size": config["batch_size"],
                "d_model": config["d_model"],
                "dropout": config["dropout"],
                "hidden_fc1_choice": config["hidden_fc1_choice"],
                "instrument_emb": config["instrument_embedding_dim"],
                "learning_rate": config["learning_rate"],
                "n_layers": config["n_layers"],
                "optimizer_name": config["optimizer_name"],
                "acquisition": acquisition,
                "budget": budget,
                "cost": cost,
                "error": error
            }
            new_records.append(record)

    df_new = pd.DataFrame(new_records)

    if not df_existing.empty:
        # Create an empty row with same columns
        empty_row = pd.DataFrame([[""] * len(df_existing.columns)], columns=df_existing.columns)
        # Combine existing, empty row, and new records
        df_combined = pd.concat([df_existing, empty_row, df_new], ignore_index=True)
    else:
        # If there was no existing CSV, this is the first write
        df_combined = df_new

    # Save updated CSV
    df_combined.to_csv(output_csv, index=False)
    print(f"CSV updated with data from {json_path} and saved to {output_csv}")


# append_json_to_csv_with_gap("/Users/madinabekbergenova/PycharmProjects/pythonProject/Scripts/smac3_output/optimization_09_04/0/runhistory.json", "/Users/madinabekbergenova/PycharmProjects/pythonProject/Optimization_results_09_04/optimization_output_devices.csv")
