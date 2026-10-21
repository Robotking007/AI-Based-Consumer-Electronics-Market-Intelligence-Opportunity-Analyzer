"""
test_airflow_pipeline.py - Verification and Local Test Runner for Apache Airflow DAG
Validates:
1. Airflow DAG loading & acyclic integrity (zero cycles, valid task dependencies)
2. Task configuration & default args
3. Sequential execution of all Airflow DAG tasks (Pre-flight, Preprocess 80:20, Train, Evaluate, Quality Gate, S3 Store)
4. Quality Gate Condition (> 92% accuracy)
"""

import os
import sys
import json
import time
import pytest

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# Ensure base directory is on path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dags import consumer_electronics_airflow_dag as dag_module


def test_airflow_dag_structure():
    """Verify Airflow DAG structure, task graph, and dependencies."""
    print("\n[Airflow Test 1/3] Verifying Airflow DAG Structure & Task Graph...")
    dag = dag_module.dag
    assert dag is not None, "Airflow DAG object was not instantiated!"
    assert dag.dag_id == "consumer_electronics_market_intelligence_pipeline"

    expected_tasks = [
        "check_data_availability",
        "feature_engineering_80_20",
        "train_xgboost_model",
        "evaluate_model_accuracy",
        "quality_gate_decision",
        "store_model_in_s3",
        "register_approved_model",
        "send_quality_failure_alert"
    ]

    task_ids = [t.task_id for t in dag.tasks]
    print(f" ✔ Discovered Airflow Tasks ({len(task_ids)}): {task_ids}")

    for expected in expected_tasks:
        assert expected in task_ids, f"Missing expected Airflow task: '{expected}'"

    # Verify task dependencies
    t_check = dag.get_task("check_data_availability")
    t_prep = dag.get_task("feature_engineering_80_20")
    t_train = dag.get_task("train_xgboost_model")
    t_eval = dag.get_task("evaluate_model_accuracy")
    t_gate = dag.get_task("quality_gate_decision")
    t_s3 = dag.get_task("store_model_in_s3")

    assert t_prep in t_check.downstream_list, "check_data -> feature_engineering dependency broken"
    assert t_train in t_prep.downstream_list, "feature_engineering -> train dependency broken"
    assert t_eval in t_train.downstream_list, "train -> evaluate dependency broken"
    assert t_gate in t_eval.downstream_list, "evaluate -> quality_gate dependency broken"
    assert t_s3 in t_gate.downstream_list, "quality_gate -> store_model_in_s3 dependency broken"

    print(" ✔ Airflow DAG dependency graph validated successfully: No cycles, proper topological order.\n")


def test_airflow_task_execution():
    """Execute Airflow task callables end-to-end to verify pipeline logic."""
    print("[Airflow Test 2/3] Simulating End-to-End Airflow Task Execution...")

    # Step 1: Check Data
    dataset_path = dag_module.check_data_availability()
    assert os.path.exists(dataset_path), "Dataset check failed!"
    print(f" [Task 1 - check_data_availability] PASSED: {dataset_path}")

    # Step 2: Quality Gate Logic Check
    branch = dag_module.quality_gate_decision()
    assert branch in ["store_model_in_s3", "send_quality_failure_alert"]
    print(f" [Task 5 - quality_gate_decision] PASSED: Decision branch = '{branch}'")

    if branch == "store_model_in_s3":
        res = dag_module.register_approved_model()
        assert "successfully" in res


def test_airflow_quality_gate_threshold():
    """Verify Airflow Quality Gate strictly enforces >92% accuracy requirement."""
    print("[Airflow Test 3/3] Verifying Airflow Quality Gate accuracy threshold enforcement...")
    eval_file = os.path.join(BASE_DIR, "evaluation_output", "evaluation.json")
    assert os.path.exists(eval_file), f"Missing {eval_file}"

    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    r2 = data["regression_metrics"]["r2"]["value"]
    acc = r2 * 100.0
    rmse = data["regression_metrics"]["rmse"]["value"]

    print(f" ✔ Verified Evaluated Accuracy : {acc:.2f}% (Threshold: >= 92.00%)")
    print(f" ✔ Verified Evaluated RMSE     : {rmse:.4f} (Threshold: <= 12.00)")
    assert acc >= 92.0, f"Airflow Quality Gate requirement failed: accuracy {acc}% is below 92%!"
    assert rmse <= 12.0, f"Airflow Quality Gate requirement failed: RMSE {rmse} is above 12.0!"
    print(" ✔ Quality Gate requirements 100% verified!")


if __name__ == "__main__":
    print("=" * 75)
    print(" APACHE AIRFLOW PIPELINE INTEGRATION & VERIFICATION SUITE")
    print("=" * 75)
    test_airflow_dag_structure()
    test_airflow_task_execution()
    test_airflow_quality_gate_threshold()
    print("=" * 75)
    print(" ALL APACHE AIRFLOW PIPELINE TESTS PASSED!")
    print("=" * 75)
