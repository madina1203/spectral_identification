import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder # Or OneHotEncoder
# Load your data (make sure 'your_optimization_results.csv' is the correct path)
try:
    df = pd.read_csv('/Users/madinabekbergenova/PycharmProjects/pythonProject/Scripts/optimization_cluster_final_12_05.csv')
except FileNotFoundError:
    print("Error: 'your_optimization_results.csv' not found. Please check the file path.")
    exit()

# 1. Sort by validation loss
# Ensure 'validation_loss' column exists
if 'cost' not in df.columns:
    print(f"Error: 'cost' column not found in the CSV. Available columns are: {df.columns.tolist()}")
    exit()

df_sorted = df.sort_values(by='cost')

print("Top 5 performing configurations:")
print(df_sorted.head())

# --- New code to save the sorted DataFrame to a new CSV file ---
output_filename = 'sorted_optimization_results_12_05.csv'
try:
    df_sorted.to_csv(output_filename, index=False) # index=False prevents pandas from writing the DataFrame index as a column
    print(f"\nSorted results saved to '{output_filename}'")
except Exception as e:
    print(f"\nError saving the CSV file: {e}")
# --- End of new code ---

# --- Analysis considering config_id ---

# 2. Correlation Analysis
# Exclude 'config_id' and potentially 'epochs' if it's not a tunable hyperparameter
columns_for_correlation = [col for col in df.columns if col not in ['config_id', 'epochs']] # Add other non-hyperparams if any
numerical_cols_for_corr = df[columns_for_correlation].select_dtypes(include=['number']).columns

if not df[numerical_cols_for_corr].empty:
    correlation_matrix = df[numerical_cols_for_corr].corr()
    # Focus on correlation with cost
    cost_corr = correlation_matrix['cost'].sort_values(ascending=False)
    print("\nCorrelation with cost (excluding config_id):")
    print(cost_corr)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f") # fmt adds formatting to annotations
    plt.title('Correlation Matrix (Numerical Hyperparameters and Cost)')
    plt.show()
else:
    print("\nNo numerical columns found for correlation analysis after exclusions.")


# 3. Visualizations (Scatter, Box plots - generally 'config_id' would not be plotted against loss)
# Example: Scatter plot for a specific hyperparameter (learning_rate vs. cost)
if 'learning_rate' in df.columns and 'cost' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='learning_rate', y='cost')
    plt.title('Learning Rate vs. Cost')
    plt.xlabel('Learning Rate')
    plt.ylabel('Cost')
    if df['learning_rate'].min() > 0 : # Apply log scale if appropriate
         plt.xscale('log')
    plt.show()

# Example: Box plot for a categorical hyperparameter (optimizer vs. cost)
if 'optimizer_name' in df.columns and 'cost' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='optimizer_name', y='cost')
    plt.title('Optimizer vs. Cost')
    plt.show()


# 4. Hyperparameter Importance Plots (Model-Based)
df_encoded = df.copy()

# Identify categorical features (excluding 'config_id' and the target 'validation_loss')
categorical_features = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
# If 'optimizer' or other known categoricals are not object type, add them manually.

# Apply Label Encoding or One-Hot Encoding to categorical features
for col in categorical_features:
    if col in df_encoded.columns: # Check if column exists
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str)) # astype(str) handles potential mixed types or NaNs gracefully

# Define features (X) and target (y)
# Exclude 'config_id', 'validation_loss', and other non-hyperparameter columns like 'epochs'
# 'epochs' might be a result from Hyperband (varying per trial) rather than a fixed hyperparameter you set.
# If 'epochs' IS a hyperparameter you set, then include it.
hyperparameter_columns = [col for col in df_encoded.columns if col not in ['config_id', 'cost', 'error', 'acquisition', 'budget']] # Exclude non-hyperparameters

if not hyperparameter_columns:
    print("\nError: No hyperparameter columns identified for feature importance analysis. Check column names and exclusions.")
else:
    X = df_encoded[hyperparameter_columns]
    y = df_encoded['cost']

    # Ensure no NaN values in features or target after encoding
    X = X.fillna(X.median()) # Or use a more sophisticated imputer
    y = y.fillna(y.median()) # Impute NaNs in target if any

    if not X.empty and not y.empty and X.shape[0] > 1 and X.shape[1] > 0: # Check for valid data
        model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True) # Added n_estimators and oob_score
        model.fit(X, y)

        print(f"\nRandom Forest OOB Score: {model.oob_score_:.4f}") # Out-of-Bag score as a quick performance check

        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': hyperparameter_columns, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

        plt.figure(figsize=(10, max(6, len(hyperparameter_columns) * 0.5))) # Adjust height for readability
        sns.barplot(data=feature_importance_df, x='importance', y='feature')
        plt.title('Hyperparameter Importances for Predicting Cost (from Random Forest)')
        plt.tight_layout() # Adjust layout
        plt.show()

        print("\nFeature Importances (most to least):")
        print(feature_importance_df)
    else:
        print("\nNot enough valid data or features to train the importance model after preprocessing.")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"Hyperparameter columns considered: {hyperparameter_columns}")
