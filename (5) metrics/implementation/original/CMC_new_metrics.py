import os
import json
import pandas as pd
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging

########################
# JSON Processing & Aggregation
########################

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def process_all_subfolders(root_folder):
    """
    Recursively process all JSON files in subfolders of `root_folder`
    and return a DataFrame where each Task Name has its average score.
    """
    logger.info("Processing folder: %s", root_folder)

    if not os.path.exists(root_folder):
        logger.error("Folder does not exist: %s", root_folder)
        return pd.DataFrame()

    task_scores = defaultdict(list)  # Collect scores for each task


    # Walk through all subfolders and files
    for subdir, _, files in os.walk(root_folder):
        logger.info("Checking %s — %d files", subdir, len(files))
        for file_name in files:
            if file_name.endswith(".json"):
                json_path = os.path.join(subdir, file_name)
                logger.info("Found JSON: %s", json_path)
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            data = [data]  # Ensure data is a list of dictionaries
                        for item in data:
                            if isinstance(item, dict) and "Task Name" in item and "Score" in item:
                                try:
                                    task_name = item["Task Name"]
                                    score = float(item["Score"])  # Ensure Score is numeric
                                    task_scores[task_name].append(score)  # Collect scores for the task
                                    logger.info("Added score for task '%s': %f", task_name, score)
                                except ValueError:
                                    logger.warning("Non-numeric score in file %s: %s", json_path, item)
                            else:
                                logger.warning("Skipping invalid entry in file %s: %s", json_path, item)
                except Exception as e:
                    logger.error("Failed to load %s: %s", json_path, e)

    if not task_scores:
        logger.warning("No valid data found in this folder or subfolders!")
        return pd.DataFrame()

    # Aggregate scores by calculating the average for each task
    aggregated_data = [
        {"Task Name": task, "Score": sum(scores) / len(scores)}  # Rename column to 'Score'
        for task, scores in task_scores.items()
    ]

    # Convert aggregated data into a DataFrame
    df = pd.DataFrame(aggregated_data)
    logger.info("Created aggregated DataFrame with %d rows", len(df))
    logger.debug("Aggregated DataFrame content:\n%s", df)

    # Save the aggregated data to a CSV file
    output_csv = os.path.join(root_folder, "aggregated_task_scores.csv")
    df.to_csv(output_csv, index=False, encoding="utf-8")
    logger.info("CSV created at: %s", output_csv)

    return df


def process_all_folders_in_parallel(root_folder):
    subfolders = [f for f in Path(root_folder).iterdir() if f.is_dir()]
    with ThreadPoolExecutor() as executor:
        dfs = list(executor.map(process_all_subfolders, subfolders))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def average_compliance(df, tasks, metric_name="Unknown Metric"):
    logger.info("Calculating average compliance for metric: %s", metric_name)
    relevant_df = df[df["Task Name"].isin(tasks)]

    zero_score_tasks = relevant_df[relevant_df["Score"] == 0]["Task Name"].tolist()
    if zero_score_tasks:
        logger.warning("The following tasks were excluded (Score = 0): %s", zero_score_tasks)

    relevant_df = relevant_df[relevant_df["Score"] != 0]

    if relevant_df.empty:
        logger.warning("No valid (non-zero) tasks found for average_compliance. Returning 0.0")
        return 0.0

    logger.info("Tasks used in average_compliance:\n%s", relevant_df[["Task Name", "Score"]])
    return relevant_df["Score"].mean()


def absolute_compliance(df, tasks, metric_name="Unknown Metric"):
    logger.info("Calculating absolute compliance for metric: %s", metric_name)
    relevant_df = df[df["Task Name"].isin(tasks)]

    zero_score_tasks = relevant_df[relevant_df["Score"] == 0]["Task Name"].tolist()
    if zero_score_tasks:
        logger.warning("The following tasks were excluded (Score = 0): %s", zero_score_tasks)

    relevant_df = relevant_df[relevant_df["Score"] != 0]

    if relevant_df.empty:
        logger.warning("No valid (non-zero) tasks found for absolute_compliance. Returning None")
        return None

    logger.info("Tasks used in absolute_compliance:\n%s", relevant_df[["Task Name", "Score"]])
    return relevant_df["Score"]


def scale_result(avg, scale="percentage", total_points=None):
    """
    Scale the average according to the desired output unit and return a string with unit.
    """
    logger.info("Scaling result: avg=%f, scale=%s, total_points=%s", avg, scale, total_points)
    if scale == "percentage":
        return f"{round((avg / 5) * 100, 2)} %"
    elif scale == "score_0_5":
        return f"{round(avg, 2)} / 5"
    elif scale == "mean":
        return f"{round(avg, 2)} (mean)"
    elif scale == "points":
        if total_points is None:
            logger.error("total_points must be provided when using scale='points'")
            raise ValueError("total_points must be provided when using scale='points'")
        return f"{round(avg, 2)} / {total_points} points"
    else:
        logger.error("Unknown scale '%s'. Use 'percentage', 'score_0_5', 'min_scores' or 'points'.", scale)
        raise ValueError(f"Unknown scale '{scale}'. Use 'percentage', 'score_0_5', 'min_scores' or 'points'.")


##### OVERALL METRICS BASED ON STAGE####

def data_ingestion_and_data_versioning_score(df, scale="percentage"):
    tasks = [
        "Data Collection", "Data Extraction", "Data Integration", "Streaming Data Handling", "Batch Data Handling",
        "Data Labeling", "Metadata Management", "Data Versioning", "Schema Validation", "Data Storage Strategy"
    ]
    avg = average_compliance(df, tasks, metric_name="data_ingestion_and_data_versioning_score")
    # There are 10 tasks for this stage, so divide by 10
    return scale_result(avg, scale)  # assuming scores are from 0 to 1


def data_preprocessing_score(df, scale="percentage"):
    tasks = [
        "Remove Duplicates", "Fix Corrupt Records", "Handle Missing Values", "Data Normalization/Scaling",
        "Feature Engineering", "Feature Selection", "Data Encoding", "Handling Imbalanced Data",
        "Data Augmentation", "Data Transformation", "Anonymization and Privacy Handling", "Data Splitting",
        "Outlier Detection and Treatment", "Data Quality Checks"
    ]
    avg = average_compliance(df, tasks, metric_name="data_preprocessing_score")
    return scale_result(avg, scale)  # 14 tasks


def model_training_and_tuning_score(df, scale="percentage"):
    tasks = [
        "Model Selection", "Defining Loss Functions and Optimizers", "Hyperparameter Initialization",
        "Data Pipeline Optimization", "Model Training", "Hyperparameter Optimization", "Cross-validation",
        "Regularization Techniques", "Parallelization and Distributed Training", "Handling Class Imbalances",
        "Explainability Features"
    ]
    avg = average_compliance(df, tasks, metric_name="model_training_and_tuning_score")
    return scale_result(avg, scale)  # 11 tasks


def model_evaluation_score(df, scale="percentage"):
    tasks = [
        "Performance Metric Calculation", "Confusion Matrix Analysis", "Precision-Recall and ROC Curve Analysis",
        "Error Analysis", "Bias and Fairness Testing", "Interpretability Analysis", "Benchmarking Against Baselines",
        "Model A/B Testing", "Uncertainty Estimation", "Cross-Validation", "Model Validation"
    ]
    avg = average_compliance(df, tasks, metric_name="model_evaluation_score")
    return scale_result(avg, scale)  # 11 tasks


def model_versioning_score(df, scale="percentage"):
    tasks = [
        "Model Serialization", "Experiment Tracking", "Model Hashing", "Model Registry Management",
        "Deployment Compatibility Testing", "Version Comparison"
    ]
    avg = average_compliance(df, tasks, metric_name="model_versioning_score")
    return scale_result(avg, scale)  # 6 tasks


def model_deployment_score(df, scale="percentage"):
    tasks = [
        "Infrastructure Setup", "Containerization", "API Integration", "Serverless Deployment",
        "Scalability Planning", "Security Measures", "Deployment A/B Testing", "Latency Optimization",
        "Continuous Integration/Continuous Deployment (CI/CD)"
    ]
    avg = average_compliance(df, tasks, metric_name="model_deployment_score")
    return scale_result(avg, scale)  # 9 tasks


def monitoring_and_maintenance_score(df, scale="percentage"):
    tasks = [
        "Performance Monitoring", "Data Drift Detection", "Concept Drift Detection", "Model Retraining Scheduling",
        "Anomaly Detection", "Logging and Debugging", "Incident Response Planning", "Resource Monitoring"
    ]
    avg = average_compliance(df, tasks, metric_name="monitoring_and_maintenance_score")
    return scale_result(avg, scale)  # 8 tasks


###################################OTher

def data_acquisition_ingestion_quality(df, scale="percentage"):
    tasks = ["Data Collection", "Data Extraction", "Streaming Data Handling", "Batch Data Handling"]
    avg = average_compliance(df, tasks, metric_name="data_acquisition_ingestion_quality")
    return scale_result(avg, scale)

def data_integration_consistency(df, scale="percentage"):
    tasks = ["Data Integration", "Metadata Management", "Schema Validation", "Data Storage Strategy", "Data Quality Checks"]
    avg = average_compliance(df, tasks, metric_name="data_integration_consistency")
    return scale_result(avg, scale)

def data_cleaning_preprocessing_robustness(df, scale="percentage"):
    tasks = [
        "Remove Duplicates", "Fix Corrupt Records", "Handle Missing Values",
        "Data Normalization/Scaling", "Data Transformation", "Data Encoding",
        "Outlier Detection and Treatment"
    ]
    avg = average_compliance(df, tasks, metric_name="data_cleaning_preprocessing_robustness")
    return scale_result(avg, scale)

def annotation_labeling_quality(df, metric_name="Annotation Labeling Quality", scale="points", total_points=None):
    """
    Compute the Annotation Labeling Quality metric using a points-based system.
    The total points dynamically adjust if some tasks are irrelevant (score = 0),
    and the final result is formatted using the scale_result() helper.
    """

    tasks = ["Data Labeling", "Handling Imbalanced Data", "Data Augmentation"]

    # Default total points (5 per task)
    if total_points is None:
        total_points = len(tasks) * 5  

    # Filter relevant tasks
    relevant_df = df[df["Task Name"].isin(tasks)]
    logger.info("=== Metric: %s ===", metric_name)
    
    if relevant_df.empty:
        logger.warning("No relevant tasks found. Returning 0 points")
        return scale_result(0, scale=scale, total_points=total_points)

    # Handle zero-score (irrelevant) tasks
    zero_score_tasks = relevant_df[relevant_df["Score"] == 0]["Task Name"].tolist()
    if zero_score_tasks:
        logger.warning("The following tasks were excluded (Score = 0):")
        for task in zero_score_tasks:
            logger.warning("   - %s", task)

    # Keep only tasks with non-zero scores
    relevant_df = relevant_df[relevant_df["Score"] != 0]

    if relevant_df.empty:
        logger.warning("All relevant tasks scored 0. Returning 0 points")
        return scale_result(0, scale=scale, total_points=total_points)

    logger.info("Tasks used:")
    logger.info("\n%s", relevant_df[["Task Name", "Score"]])

    # Dynamically adjust total_points based on active (non-zero) tasks
    base_points_per_task = total_points / len(tasks)
    dynamic_total_points = base_points_per_task * len(relevant_df)

    total_score = relevant_df["Score"].sum()
    total_score = min(total_score, dynamic_total_points)

    # Use the scaling function for consistent output
    return scale_result(total_score, scale=scale, total_points=dynamic_total_points)





def data_governance_reproducibility(df, scale ="score_0_5"):
    tasks = ["Data Versioning", "Model Hashing", "Experiment Tracking", "Model Registry Management", "Metadata Management"]
    avg = average_compliance(df, tasks)
    return scale_result(avg, scale)

def privacy_ethical_data_handling(df, metric_name="Privacy & Ethical Data Handling", scale="points", total_points=None):
    """
    Compute the Privacy & Ethical Data Handling metric using a points-based system.
    The total points dynamically adjust if some tasks are irrelevant (score = 0),
    and the final result is formatted using the scale_result() helper.
    """

    tasks = ["Anonymization and Privacy Handling", "Bias and Fairness Testing", "Security Measures"]

    # Default total points (5 per task)
    if total_points is None:
        total_points = len(tasks) * 5  

    # Filter relevant tasks
    relevant_df = df[df["Task Name"].isin(tasks)]
    logger.info("=== Metric: %s ===", metric_name)
    
    if relevant_df.empty:
        logger.warning("No relevant tasks found. Returning 0 points")
        return scale_result(0, scale=scale, total_points=total_points)

    # Handle zero-score (irrelevant) tasks
    zero_score_tasks = relevant_df[relevant_df["Score"] == 0]["Task Name"].tolist()
    if zero_score_tasks:
        logger.warning("The following tasks were excluded (Score = 0):")
        for task in zero_score_tasks:
            logger.warning("   - %s", task)

    # Keep only tasks with non-zero scores
    relevant_df = relevant_df[relevant_df["Score"] != 0]

    if relevant_df.empty:
        logger.warning("All relevant tasks scored 0. Returning 0 points")
        return scale_result(0, scale=scale, total_points=total_points)

    logger.info("Tasks used:")
    logger.info("\n%s", relevant_df[["Task Name", "Score"]])

    # Dynamically adjust total_points based on active (non-zero) tasks
    base_points_per_task = total_points / len(tasks)
    dynamic_total_points = base_points_per_task * len(relevant_df)

    total_score = relevant_df["Score"].sum()
    total_score = min(total_score, dynamic_total_points)

    # Use the scaling function for consistent output
    return scale_result(total_score, scale=scale, total_points=dynamic_total_points)




def feature_engineering_selection_efficiency(df, scale="percentage"):
    tasks = ["Feature Engineering", "Feature Selection", "Data Encoding", "Data Transformation"]
    avg = average_compliance(df, tasks, metric_name="feature_engineering_selection_efficiency")
    return scale_result(avg, scale)

def model_training_optimization(df, scale="percentage"):
    tasks = [
        "Model Training", "Hyperparameter Initialization", "Hyperparameter Optimization",
        "Regularization Techniques", "Parallelization and Distributed Training",
        "Handling Class Imbalances", "Data Pipeline Optimization"
    ]
    avg = average_compliance(df, tasks, metric_name="model_training_optimization")
    return scale_result(avg, scale)

def model_validation_evaluation(df, scale="percentage"):
    tasks = [
        "Model Validation", "Cross-Validation", "Performance Metric Calculation",
        "Confusion Matrix Analysis", "Precision-Recall and ROC Curve Analysis",
        "Error Analysis", "Uncertainty Estimation", "Benchmarking Against Baselines"
    ]
    avg = average_compliance(df, tasks, metric_name="model_validation_evaluation")
    return scale_result(avg, scale)

def model_explainability_interpretability(df, scale="percentage"):
    tasks = ["Interpretability Analysis", "Explainability Features"]
    avg = average_compliance(df, tasks, metric_name="model_explainability_interpretability")
    return scale_result(avg, scale)

def deployment_readiness_automation(df, scale="percentage"):
    tasks = [
        "Infrastructure Setup", "Containerization", "API Integration", "Serverless Deployment",
        "Scalability Planning", "Deployment A/B Testing", "Continuous Integration/Continuous Deployment (CI/CD)",
        "Model Serialization", "Deployment Compatibility Testing"
    ]
    avg = average_compliance(df, tasks, metric_name="deployment_readiness_automation")
    return scale_result(avg, scale)

def security_incident_management(df, scale="percentage"):
    tasks = ["Security Measures", "Incident Response Planning", "Logging and Debugging", "Resource Monitoring"]
    avg = average_compliance(df, tasks, metric_name="security_incident_management")
    return scale_result(avg, scale)

def monitoring_continuous_improvement(df, scale="percentage"):
    tasks = ["Performance Monitoring", "Data Drift Detection", "Concept Drift Detection", "Model Retraining Scheduling", "Anomaly Detection"]
    avg = average_compliance(df, tasks, metric_name="monitoring_continuous_improvement")
    return scale_result(avg, scale)


def auditability_index(df, scale="score_0_5"):
    tasks = ["Metadata Management", "Data Versioning", "Logging and Debugging", "Model Registry Management", "Experiment Tracking"]

    scores = absolute_compliance(df, tasks, metric_name="auditability_index")
    if scores is None:
        return 0.0  # no valid tasks to score

    # Recalculate weight based on number of valid tasks
    weight = 1 / len(scores)
    raw_score = sum(weight * scores)

    # ✅ Apply scaling to final value, just like other metrics
    return scale_result(raw_score, scale)

def data_integrity_score(df, scale="score_0_5"):
    tasks = ["Data Quality Checks", "Schema Validation"]
    relevant_df = df[df["Task Name"].isin(tasks)]

    if relevant_df.empty:
        logger.warning("No relevant tasks found for Data Integrity Score. Returning 0.")
        return 0.0  # No relevant tasks, score is 0

    # Get the minimum score across the relevant tasks
    min_score = relevant_df["Score"].min()
    logger.info("Relevant scores for Data Integrity Score:\n%s", relevant_df["Score"])

    # ✅ Apply scaling to the result
    return scale_result(min_score, scale)

def auditability_ethical_robustness_index(df, scale="score_0_5"): 
    tasks = [
        "Metadata Management", "Data Versioning", "Schema Validation", "Data Quality Checks",
        "Performance Metric Calculation", "Bias and Fairness Testing", "Interpretability Analysis",
        "Model Validation", "Experiment Tracking", "Model Registry Management",
        "Performance Monitoring", "Data Drift Detection", "Logging and Debugging"
    ]
    
    # Filter for relevant tasks
    relevant_df = df[df["Task Name"].isin(tasks)]

    # Optionally filter out zeros (if you don't want them to skew the median)
    relevant_df = relevant_df[relevant_df["Score"] != 0]

    if relevant_df.empty:
        logger.warning("No relevant tasks found for Auditability Ethical Robustness Index. Returning 0.")
        return 0.0  # No relevant tasks found

    median_score = relevant_df["Score"].median()
    logger.info("Relevant scores for Auditability Ethical Robustness Index:\n%s", relevant_df["Score"])

    # ✅ Apply scaling to the final result
    return scale_result(round(median_score, 2), scale)




##### NEW ADDED METRICS ############################


def accountability_readiness_index(df, scale="score_0_5"):
    tasks = [
        "Performance Metric Calculation", "Bias and Fairness Testing", "Interpretability Analysis",
        "Error Analysis", "Performance Monitoring", "Data Drift Detection", 
        "Incident Response Planning", "Logging and Debugging", "Model Retraining Scheduling"
    ]

    relevant_df = df[df["Task Name"].isin(tasks)]
    if relevant_df.empty:
        return scale_result(0, scale)

    # Exclude zero-score tasks
    relevant_df = relevant_df[relevant_df["Score"] != 0]
    if relevant_df.empty:
        return scale_result(0, scale)

    avg_score = relevant_df["Score"].mean()
    proportion_max = (relevant_df["Score"] == 5).sum() / len(relevant_df)
    score = avg_score * proportion_max * 5  # Multiply by 5 to keep scale compatible

    return scale_result(score, scale)


def accountability_robustness_score(df, scale="score_0_5"):
    tasks = [
        "Performance Metric Calculation", "Bias and Fairness Testing", "Interpretability Analysis",
        "Error Analysis", "Performance Monitoring", "Data Drift Detection", 
        "Incident Response Planning", "Logging and Debugging", "Model Retraining Scheduling"
    ]

    relevant_df = df[df["Task Name"].isin(tasks)]
    if relevant_df.empty:
        return scale_result(0, scale)

    # Exclude zero-score tasks
    relevant_df = relevant_df[relevant_df["Score"] != 0]
    if relevant_df.empty:
        return scale_result(0, scale)

    score = relevant_df["Score"].median() - relevant_df["Score"].std()
    return scale_result(score, scale)


def accountability_balance_score(df, scale="score_0_5"):
    # Define pre/post-deployment task grouping
    pre_tasks = ["Performance Metric Calculation", "Bias and Fairness Testing", "Interpretability Analysis", "Error Analysis"]
    post_tasks = ["Performance Monitoring", "Data Drift Detection", "Incident Response Planning", "Logging and Debugging", "Model Retraining Scheduling"]

    pre_df = df[df["Task Name"].isin(pre_tasks)]
    post_df = df[df["Task Name"].isin(post_tasks)]

    pre_sum = pre_df["Score"].sum() if not pre_df.empty else 0
    post_sum = post_df["Score"].sum() if not post_df.empty else 0

    diff = pre_sum - post_sum

    # Min-max normalize between -1 and 1
    max_abs = max(abs(diff), 1)  # prevent division by zero
    score = diff / max_abs

    return scale_result(score, scale)


def accountability_interaction_index(df, scale="score_0_5"):
    tasks = ["Bias and Fairness Testing", "Interpretability Analysis", "Incident Response Planning"]
    relevant_df = df[df["Task Name"].isin(tasks)]
    if relevant_df.empty:
        return scale_result(0, scale)

    # Exclude zero-score tasks
    relevant_df = relevant_df[relevant_df["Score"] != 0]
    if relevant_df.empty:
        return scale_result(0, scale)

    score = relevant_df["Score"].prod()
    return scale_result(score, scale)


def accountability_failure_penalty_score(df, scale="score_0_5"):
    tasks = [
        "Performance Metric Calculation", "Bias and Fairness Testing", "Interpretability Analysis",
        "Error Analysis", "Performance Monitoring", "Data Drift Detection", 
        "Incident Response Planning", "Logging and Debugging", "Model Retraining Scheduling"
    ]
    relevant_df = df[df["Task Name"].isin(tasks)]
    if relevant_df.empty:
        return scale_result(0, scale)

    relevant_df = relevant_df[relevant_df["Score"] != 0]
    if relevant_df.empty:
        return scale_result(0, scale)

    score = relevant_df["Score"].mean() - (5 - relevant_df["Score"].min())
    return scale_result(score, scale)


def data_management_maturity(df, scale="score_0_5"):
    tasks = ["Metadata Management", "Data Versioning", "Data Quality Checks"]
    weights = [0.3, 0.3, 0.4]

    relevant_df = df[df["Task Name"].isin(tasks)]
    if relevant_df.empty:
        return scale_result(0, scale)

    # Keep only non-zero tasks
    relevant_df = relevant_df[relevant_df["Score"] != 0]
    if relevant_df.empty:
        return scale_result(0, scale)

    # Align weights to the filtered tasks
    task_indices = [tasks.index(task) for task in relevant_df["Task Name"].values]
    score = sum(relevant_df["Score"].values[i] * weights[task_indices[i]] for i in range(len(relevant_df)))

    return scale_result(score, scale)



def bias_fairness_interpretability_index(df, scale="score_0_5"):
    tasks = ["Bias and Fairness Testing", "Interpretability Analysis"]
    relevant_df = df[df["Task Name"].isin(tasks)]
    if relevant_df.empty:
        return scale_result(0, scale)

    relevant_df = relevant_df[relevant_df["Score"] != 0]
    if relevant_df.empty:
        return scale_result(0, scale)

    score = 0.5 * relevant_df["Score"].iloc[0] + 0.5 * relevant_df["Score"].iloc[1]
    return scale_result(score, scale)


def model_robustness_score(df, scale="score_0_5"):
    tasks = ["Model Validation", "Performance Monitoring"]
    relevant_df = df[df["Task Name"].isin(tasks)]
    if relevant_df.empty:
        return scale_result(0, scale)

    relevant_df = relevant_df[relevant_df["Score"] != 0]
    if relevant_df.empty:
        return scale_result(0, scale)

    score = relevant_df["Score"].min()
    return scale_result(score, scale)


def reproducibility_index(df, scale="score_0_5"):
    tasks = ["Experiment Tracking", "Model Registry Management", "Data Versioning"]
    weights = [0.4, 0.3, 0.3]

    relevant_df = df[df["Task Name"].isin(tasks)]
    if relevant_df.empty:
        return scale_result(0, scale)

    # Keep only non-zero tasks
    relevant_df = relevant_df[relevant_df["Score"] != 0]
    if relevant_df.empty:
        return scale_result(0, scale)

    # Align weights to filtered tasks
    task_indices = [tasks.index(task) for task in relevant_df["Task Name"].values]
    score = sum(relevant_df["Score"].values[i] * weights[task_indices[i]] for i in range(len(relevant_df)))

    return scale_result(score, scale)


# Weighted averages
def operational_transparency_avg(df, scale="score_0_5"):
    tasks = ["Logging and Debugging", "Continuous Integration/Continuous Deployment"]
    avg = average_compliance(df, tasks, metric_name="operational_transparency_avg")
    return scale_result(avg, scale)

def accountability_readiness_weighted_avg(df, scale="score_0_5"):
    tasks = [
        "Bias and Fairness Testing", "Error Analysis", "Anomaly Detection", "Incident Response Planning",
        "Logging and Debugging", "Performance Monitoring", "Data Drift Detection", "Concept Drift Detection"
    ]
    weights = [0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05]
    avg = average_compliance(df, tasks, metric_name="accountability_readiness_weighted_avg", weights=weights)
    return scale_result(avg, scale)

# Min/Max comparisons
def accountability_stability_min_max(df, scale="score_0_5"):
    tasks = [
        "Bias and Fairness Testing", "Error Analysis", "Anomaly Detection", "Incident Response Planning",
        "Logging and Debugging", "Performance Monitoring", "Data Drift Detection", "Concept Drift Detection"
    ]
    relevant_df = df[df["Task Name"].isin(tasks)]
    if relevant_df.empty:
        return scale_result(0, scale)
    min_score = relevant_df["Score"].min()
    max_score = relevant_df["Score"].max()
    ratio = min_score / max_score if max_score != 0 else 0
    return scale_result(ratio * 5, scale)  # scale to 0-5 for score_0_5

# Proportions / thresholds
def accountability_risk_proportion(df, threshold=3, scale="percentage"):
    tasks = [
        "Bias and Fairness Testing", "Error Analysis", "Anomaly Detection", "Incident Response Planning",
        "Logging and Debugging", "Performance Monitoring", "Data Drift Detection", "Concept Drift Detection"
    ]
    relevant_df = df[df["Task Name"].isin(tasks)]
    if relevant_df.empty:
        return scale_result(0, scale)
    proportion = (relevant_df["Score"] < threshold).mean()
    return scale_result(proportion * 5, scale)

def ethical_compliance_threshold_proportion(df, threshold=4, scale="percentage"):
    tasks = [
        "Bias and Fairness Testing", "Interpretability Analysis", "Model Selection",
        "Defining Loss Functions and Optimizers", "Hyperparameter Optimization",
        "Performance Monitoring", "Data Drift Detection", "Incident Response Planning"
    ]
    relevant_df = df[df["Task Name"].isin(tasks)]
    if relevant_df.empty:
        return scale_result(0, scale)
    proportion = (relevant_df["Score"] >= threshold).mean()
    return scale_result(proportion * 5, scale)

# Standard deviation / variance
def accountability_variance_std(df, scale="score_0_5"):
    tasks = [
        "Bias and Fairness Testing", "Error Analysis", "Anomaly Detection", "Incident Response Planning",
        "Logging and Debugging", "Performance Monitoring", "Data Drift Detection", "Concept Drift Detection"
    ]
    relevant_df = df[df["Task Name"].isin(tasks)]
    if relevant_df.empty:
        return scale_result(0, scale)
    std_dev = relevant_df["Score"].std()
    return scale_result(std_dev, scale)

def ethical_variance_std(df, scale="score_0_5"):
    tasks = [
        "Bias and Fairness Testing", "Interpretability Analysis", "Model Selection",
        "Defining Loss Functions and Optimizers", "Hyperparameter Optimization",
        "Performance Monitoring", "Data Drift Detection", "Incident Response Planning"
    ]
    relevant_df = df[df["Task Name"].isin(tasks)]
    if relevant_df.empty:
        return scale_result(0, scale)
    std_dev = relevant_df["Score"].std()
    return scale_result(std_dev, scale)

# Product / interaction
def accountability_interaction_product(df, scale="score_0_5"):
    tasks = ["Bias and Fairness Testing", "Logging and Debugging"]
    relevant_df = df[df["Task Name"].isin(tasks)]
    if relevant_df.empty or len(relevant_df) < 2:
        return scale_result(0, scale)
    product = relevant_df["Score"].prod()
    return scale_result(product, scale)

def ethical_performance_interaction_product(df, scale="score_0_5"):
    tasks = [
        "Bias and Fairness Testing", "Interpretability Analysis", "Model Selection",
        "Defining Loss Functions and Optimizers", "Hyperparameter Optimization",
        "Performance Monitoring", "Data Drift Detection", "Incident Response Planning"
    ]
    relevant_df = df[df["Task Name"].isin(tasks)]
    if relevant_df.empty:
        return scale_result(0, scale)
    product = relevant_df["Score"].prod()
    return scale_result(product, scale)

# Median / robustness
def ethical_robustness_median(df, scale="score_0_5"):
    tasks = [
        "Bias and Fairness Testing", "Interpretability Analysis", "Model Selection",
        "Defining Loss Functions and Optimizers", "Hyperparameter Optimization",
        "Performance Monitoring", "Data Drift Detection", "Incident Response Planning"
    ]
    relevant_df = df[df["Task Name"].isin(tasks)]
    if relevant_df.empty:
        return scale_result(0, scale)
    median_score = relevant_df["Score"].median()
    return scale_result(median_score, scale)









##### NEW ADDED METRICS ############################


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate JSON scores from multiple subfolders into average scores per task."
    )
    parser.add_argument(
        "--root", "-r",
        type=str,
        default="compliance_results",
        help="Root folder containing subfolders (default: compliance_results)."
    )
    parser.add_argument(
        "--folder", "-f",
        type=str,
        help="Specific first-level folder to process inside the root folder."
    )
    args = parser.parse_args()

    # Determine target folder(s)
    if args.folder:
        folder_path = os.path.join(args.root, args.folder)
        if not os.path.isdir(folder_path):
            logger.error("Folder '%s' not found inside '%s'", args.folder, args.root)
            return
        final_df = process_all_subfolders(folder_path)
    else:
        all_dfs = []
        for sub in os.listdir(args.root):
            sub_path = os.path.join(args.root, sub)
            if os.path.isdir(sub_path):
                df = process_all_subfolders(sub_path)
                if not df.empty:
                    all_dfs.append(df)
        final_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    if final_df.empty:
        logger.warning("No data processed. Exiting.")
        return

    logger.info("Aggregated DataFrame:\n%s", final_df.head())

    metrics = {
        "Data Ingestion and Versioning": data_ingestion_and_data_versioning_score(final_df),
        "Data Preprocessing": data_preprocessing_score(final_df),
        "Model Training and Tuning": model_training_and_tuning_score(final_df),
        "Model Evaluation": model_evaluation_score(final_df),
        "Model Versioning": model_versioning_score(final_df),
        "Model Deployment": model_deployment_score(final_df),
        "Monitoring and Maintenance": monitoring_and_maintenance_score(final_df),

        "Data Acquisition & Ingestion Quality": data_acquisition_ingestion_quality(final_df),
        "Data Integration & Consistency": data_integration_consistency(final_df),
        "Data Cleaning & Preprocessing Robustness": data_cleaning_preprocessing_robustness(final_df),
        "Annotation & Labeling Quality": annotation_labeling_quality(final_df),
        "Data Governance & Reproducibility": data_governance_reproducibility(final_df),
        "Privacy & Ethical Data Handling": privacy_ethical_data_handling(final_df),
        "Feature Engineering & Selection Efficiency": feature_engineering_selection_efficiency(final_df),
        "Model Training & Optimization": model_training_optimization(final_df),
        "Model Validation & Evaluation": model_validation_evaluation(final_df),
        "Model Explainability & Interpretability": model_explainability_interpretability(final_df),
        "Deployment Readiness & Automation": deployment_readiness_automation(final_df),
        "Security & Incident Management": security_incident_management(final_df),
        "Monitoring & Continuous Improvement": monitoring_continuous_improvement(final_df),
        "Auditability Index": auditability_index(final_df),
        "Data Integrity Score": data_integrity_score(final_df),
        "Auditability Ethical Robustness Index": auditability_ethical_robustness_index(final_df),

        "new":"new",
        #new
        "Accountability Readiness Index": accountability_readiness_index(final_df),
        "Accountability Robustness Score": accountability_robustness_score(final_df),
        "Accountability Balance Score": accountability_balance_score(final_df),
        "Accountability Interaction Index": accountability_interaction_index(final_df),
        "Accountability Failure Penalty Score": accountability_failure_penalty_score(final_df),

        "Data Management Maturity": data_management_maturity(final_df),
        "Bias, Fairness & Interpretability Index": bias_fairness_interpretability_index(final_df),
        "Model Robustness Score": model_robustness_score(final_df),
        "Reproducibility Index": reproducibility_index(final_df),

       
        "Privacy & Ethical Data Handling": privacy_ethical_data_handling(final_df),
        "Annotation Labeling Quality": annotation_labeling_quality(final_df),

        # Accountability / Operational Metrics
        "Operational Transparency (Avg)": operational_transparency_avg(final_df),
        "Accountability Stability (Min/Max Ratio)": accountability_stability_min_max(final_df),
        "Accountability Risk (Proportion <3)": accountability_risk_proportion(final_df),
        "Accountability Variance (Std Dev)": accountability_variance_std(final_df),
        "Accountability Interaction (Product)": accountability_interaction_product(final_df),

        # Ethical / Robustness Metrics
        "Ethical Robustness (Median)": ethical_robustness_median(final_df),
        "Ethical Compliance Threshold (Proportion >=4)": ethical_compliance_threshold_proportion(final_df),
        "Ethical Variance (Std Dev)": ethical_variance_std(final_df),
        "Ethical Performance Interaction (Product)": ethical_performance_interaction_product(final_df),
    }

    logger.info("Metrics Summary:")
    for name, value in metrics.items():
        logger.info("  %s: %s", name, value)

    return final_df


if __name__ == "__main__":
    main()

# Process specific folder inside compliance_results
#python CMC_folders.py --folder ML_GRUPO01_BEFORE

# Process another folder
#python CMC_folders.py --folder a_few_good

# Process everything under a custom root
#python CMC_folders.py --root my_results_folder



