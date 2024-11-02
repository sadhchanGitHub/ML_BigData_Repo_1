# these modules are used by v_1.1_DecisionTrees_Parallelized_GridSearch_Credit_Card_Fraud_Detection

import os
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score, precision_recall_curve, auc
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import multiprocessing as mp


# Function for parallelized one-hot encoding of categorical variables
def parallel_one_hot_encode(df, num_partitions=None):
    if num_partitions is None:
        num_partitions = mp.cpu_count()  # Use all available CPU cores

    # Split the dataframe into chunks
    df_split = np.array_split(df, num_partitions)

    # Create a multiprocessing Pool
    with mp.Pool(num_partitions) as pool:
        # Apply pd.get_dummies in parallel
        result = pool.map(pd.get_dummies, df_split)

    # Concatenate the results back into a single dataframe
    return pd.concat(result)

"""
# Function for preprocessing: one-hot encoding and imputing missing values
def preprocessdata_parallel_onehot_impute(df, selected_features):
    y = df['is_fraud']
    X = df[selected_features]

    # Parallelize one-hot encoding of categorical variables
    X_encoded = parallel_one_hot_encode(X)

    # Get column names before imputation
    feature_names = X_encoded.columns.tolist()

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_encoded)

    return X_imputed, y, feature_names
"""

# Updated function with feature names for visualization
def preprocessdata_parallel_onehot_impute(df, selected_features, sample_fraction=0.3):
    df_sampled = df.sample(frac=sample_fraction, random_state=42)
    y = df_sampled['is_fraud']
    X = df_sampled[selected_features]

    # Use one-hot encoding
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Capture feature names for visualization
    feature_names = X_encoded.columns.tolist()

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_encoded)

    return X_imputed, y, feature_names



# Function to perform GridSearchCV for hyperparameter tuning
def perform_grid_search(X, y, param_grid):
    decision_tree_model = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=decision_tree_model, param_grid=param_grid, 
                               cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    return grid_search.best_estimator_


# Update perform_grid_search to use balanced class weight
def perform_grid_search_balanced(X, y, param_grid):
    decision_tree_model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
    grid_search = GridSearchCV(estimator=decision_tree_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    return grid_search.best_estimator_



# Step 3: Generate Classification Report and ROC Curve
def generate_classification_report_and_roc(model, X, y, dataset_type, model_specs, reports_output_dir):

    # Generate a timestamp for the report filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    clf_report = classification_report(y, y_pred)
    print(f"{model_specs}_{dataset_type} Classification Report:\n", clf_report)
    

    
    # Calculate probabilities and AUC
    y_probs = model.predict_proba(X)[:, 1]
    roc_auc = roc_auc_score(y, y_probs)
    fpr, tpr, _ = roc_curve(y, y_probs)
    precision, recall, _ = precision_recall_curve(y, y_probs)
    auc_pr = auc(recall, precision)
    print(f"{model_specs}_{dataset_type} ROC AUC:", roc_auc)
    print(f"{model_specs}_{dataset_type} Precision-Recall AUC:", auc_pr)
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'{dataset_type} ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_type}')
    plt.legend(loc="lower right")
    

    plot_filename = f"{model_specs}_{dataset_type}_RoC_Curve_{timestamp}.png"
    plt.savefig(os.path.join(reports_output_dir, plot_filename))
    plt.close()
    print(f"ROC_Curve saved to {plot_filename}")
    
    # Create and save classification report
    report_filename = f"{model_specs}_{dataset_type}_Report_{timestamp}.txt"
    report_path = os.path.join(reports_output_dir, report_filename)
    full_report = (
        f"Accuracy: {accuracy:.4f}\n\n"
        f"Classification Report:\n{clf_report}\n\n"
        f"ROC AUC: {roc_auc:.4f}\n"
        f"Precision-Recall AUC: {auc_pr:.4f}"
    )
    with open(report_path, "w") as f:
        f.write(full_report)
    print(f"Classification report saved to {report_path}")    

def save_decision_tree_viz(model, feature_names, class_names, output_path):
    plt.figure(figsize=(20, 10))  # Adjust size as needed
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title("Decision Tree Gridsearch Visualization")
    plt.savefig(output_path, format="png", dpi=300)  # Save as PNG with high resolution
    plt.close()
    print(f"Decision Tree visualization saved to {output_path}")


