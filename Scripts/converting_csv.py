import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# Specify the directory containing the CSV files
input_directory = "/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/temp_for_distribution_check"
output_file = "merged.csv"
output_plot_dir = "/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/Exploris/csv_distribution"


def merge_csv_files(input_dir, output_file):
    csv_files = glob(os.path.join(input_dir, "*.csv"))
    merged_df = pd.DataFrame()

    for file in csv_files:
        df = pd.read_csv(file)
        merged_df = pd.concat([merged_df, df], ignore_index=True, sort=False)

    merged_df.to_csv(output_file, index=False)
    print(f"Merged CSV saved to: {output_file}")
    return merged_df
# Step 1: Merge CSV files
# Step 1: Merge CSV files
# Step 2: Sanitize column names for safe file paths
def sanitize_filename(name):
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name)


# Step 3: Analyze distributions and save plots
def analyze_and_save_plots(df, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)  # Create the directory if it doesn't exist

    for column in df.columns:
        print(f"Saving plot for column: {column}")
        plt.figure(figsize=(8, 6))
        if df[column].dtype in ['int64', 'float64']:
            # Numeric data - Plot histogram
            df[column].dropna().plot.hist(bins=20, edgecolor='black')
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
        else:
            # Categorical data - Plot bar chart
            value_counts = df[column].value_counts()
            value_counts.plot.bar()
            plt.title(f"Value Counts of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")

        plt.grid(True)
        sanitized_column = sanitize_filename(column)
        plot_path = os.path.join(plot_dir, f"{sanitized_column}_distribution.png")
        plt.savefig(plot_path)  # Save the plot as a file
        plt.close()  # Close the plot to free memory
        print(f"Plot saved: {plot_path}")


# Main script
if __name__ == "__main__":
    merged_df = merge_csv_files(input_directory, output_file)
    analyze_and_save_plots (merged_df,output_plot_dir)
    print(f"All plots saved in directory: {output_plot_dir}")
