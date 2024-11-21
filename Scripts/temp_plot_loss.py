import pandas as pd
import matplotlib.pyplot as plt

# List of CSV files and corresponding labels for the plot
csv_files = [
    'csv_logs/spectra_transformer_experiment/version_25/metrics.csv',

]
labels = ['Version 25']

# Initialize dictionaries to store metrics from all versions
metrics_dict = {
    'train_loss': [],
    'val_loss': [],

    'train_f1': [],
    'val_f1': [],
    'train_recall': [],
    'val_recall': []
}

# Load data from each CSV file
for csv_file, label in zip(csv_files, labels):
    metrics = pd.read_csv(csv_file)

    # Extract and store metrics for each version
    metrics_dict['train_loss'].append((metrics[metrics['train_loss'].notnull()], label))
    metrics_dict['val_loss'].append((metrics[metrics['val_loss'].notnull()], label))

    metrics_dict['train_f1'].append((metrics[metrics['train_f1'].notnull()], label))
    metrics_dict['val_f1'].append((metrics[metrics['val_f1'].notnull()], label))
    metrics_dict['train_recall'].append((metrics[metrics['train_recall'].notnull()], label))
    metrics_dict['val_recall'].append((metrics[metrics['val_recall'].notnull()], label))

# Plot Training and Validation Loss
plt.figure(figsize=(10, 5))
for data, label in metrics_dict['train_loss']:
    plt.plot(data['epoch'], data['train_loss'], marker='o', label=f'{label} - Train Loss')
for data, label in metrics_dict['val_loss']:
    plt.plot(data['epoch'], data['val_loss'], marker='o', linestyle='--', label=f'{label} - Val Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Across Versions')
plt.legend()
plt.grid(True)
plt.show()

# # Plot Training and Validation Precision
# plt.figure(figsize=(10, 5))
#
# plt.xlabel('Epoch')
# plt.ylabel('Precision')
# plt.title('Training and Validation Precision Across Versions')
# plt.legend()
# plt.grid(True)
# plt.show()

# Plot Training and Validation F1 Score
plt.figure(figsize=(10, 5))
for data, label in metrics_dict['train_f1']:
    plt.plot(data['epoch'], data['train_f1'], marker='o', label=f'{label} - Train F1 Score')
for data, label in metrics_dict['val_f1']:
    plt.plot(data['epoch'], data['val_f1'], marker='o', linestyle='--', label=f'{label} - Val F1 Score')

plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('Training and Validation F1 Score Across Versions')
plt.legend()
plt.grid(True)
plt.show()

# Plot Training and Validation Recall
plt.figure(figsize=(10, 5))
for data, label in metrics_dict['train_recall']:
    plt.plot(data['epoch'], data['train_recall'], marker='o', label=f'{label} - Train Recall')
for data, label in metrics_dict['val_recall']:
    plt.plot(data['epoch'], data['val_recall'], marker='o', linestyle='--', label=f'{label} - Val Recall')

plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Training and Validation Recall Across Versions')
plt.legend()
plt.grid(True)
plt.show()
