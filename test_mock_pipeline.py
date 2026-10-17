"""
test_mock_pipeline.py - Automated Test Suite for Local SageMaker Pipeline Simulator
Validates:
1. SageMaker Pipeline Schema (Workflow Schema 2020-12-01)
2. Local Mock Pipeline End-to-End Execution (Processing -> Training -> Evaluation -> Quality Gate)
3. Quality Gate Approval Branching (RMSE <= Threshold -> RegisterModel)
4. Quality Gate Rejection Branching (RMSE > Threshold -> Block Registration)
5. Model Artifact Packaging & Inference Compatibility
"""

import os
import json
import tarfile
import pytest
import numpy as np
import xgboost as xgb
from mock_sagemaker_runner import MockSageMakerPipelineExecutor


@pytest.fixture(scope="session")
def pipeline_definition():
    def_path = "pipeline_definition.json"
    assert os.path.exists(def_path), f"Missing {def_path}"
    with open(def_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_pipeline_definition_schema(pipeline_definition):
    """Verify compliance with SageMaker Pipeline Workflow schema."""
    assert pipeline_definition["Version"] == "2020-12-01"
    assert "Parameters" in pipeline_definition
    assert "Steps" in pipeline_definition

    param_names = [p["Name"] for p in pipeline_definition["Parameters"]]
    assert "RmseThreshold" in param_names
    assert "InputDataUrl" in param_names

    step_names = [s["Name"] for s in pipeline_definition["Steps"]]
    assert "PreprocessMarketData" in step_names
    assert "TrainDemandGrowthModel" in step_names
    assert "EvaluateChallengerModel" in step_names
    assert "QualityGate-Condition" in step_names


def test_mock_pipeline_quality_gate_approval():
    """Verify full local execution when Quality Gate threshold passes."""
    executor = MockSageMakerPipelineExecutor(
        pipeline_def_path="pipeline_definition.json",
        output_root="mock_sagemaker_test_runs"
    )

    summary = executor.run(
        parameter_overrides={"RmseThreshold": 12.0},
        sample_limit=500  # Fast execution for tests
    )

    assert summary["PipelineExecutionStatus"] == "Succeeded"
    assert "exec-" in summary["PipelineExecutionDisplayName"]

    # Check that Condition Step Passed
    cond_step = next(s for s in summary["StepExecutionDetails"] if s["StepName"] == "QualityGate-Condition")
    assert cond_step["ConditionMet"] is True
    assert len(cond_step["ExecutedSubSteps"]) == 1
    assert cond_step["ExecutedSubSteps"][0]["StepName"] == "RegisterApprovedChallengerModel"

    # Check Model Artifact
    reg_meta_file = os.path.join(executor.run_dir, "registered_model_package.json")
    assert os.path.exists(reg_meta_file), "Model package file should exist upon approval"

    with open(reg_meta_file, "r", encoding="utf-8") as f:
        reg_data = json.load(f)
    assert reg_data["ModelApprovalStatus"] == "Approved"
    assert reg_data["Metrics"]["rmse"] <= 12.0


def test_mock_pipeline_quality_gate_rejection():
    """Verify Quality Gate correctly rejects promotion when threshold is breached."""
    executor = MockSageMakerPipelineExecutor(
        pipeline_def_path="pipeline_definition.json",
        output_root="mock_sagemaker_test_runs"
    )

    # Impossibly low threshold to force rejection
    summary = executor.run(
        parameter_overrides={"RmseThreshold": 0.5},
        sample_limit=500
    )

    assert summary["PipelineExecutionStatus"] == "Succeeded"

    cond_step = next(s for s in summary["StepExecutionDetails"] if s["StepName"] == "QualityGate-Condition")
    assert cond_step["ConditionMet"] is False

    reg_meta_file = os.path.join(executor.run_dir, "registered_model_package.json")
    assert not os.path.exists(reg_meta_file), "Model package must NOT be registered when Quality Gate fails"


def test_model_artifact_loadability_and_inference():
    """Verify packaged model.tar.gz can be extracted and run for live inference."""
    executor = MockSageMakerPipelineExecutor(
        pipeline_def_path="pipeline_definition.json",
        output_root="mock_sagemaker_test_runs"
    )

    summary = executor.run(
        parameter_overrides={"RmseThreshold": 12.0},
        sample_limit=500
    )

    tar_path = summary["StepOutputs"]["TrainDemandGrowthModel"]["model_tar_gz"]
    assert os.path.exists(tar_path), f"Missing model tarball at {tar_path}"

    extract_dir = os.path.join(executor.run_dir, "extracted_model_test")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    model_file = os.path.join(extract_dir, "xgboost-model.json")
    if not os.path.exists(model_file):
        model_file = os.path.join(extract_dir, "xgboost-model")
    assert os.path.exists(model_file)

    model = xgb.XGBRegressor()
    model.load_model(model_file)

    # 27 features input vector
    dummy_input = np.ones((1, 27))
    pred = model.predict(dummy_input)
    assert len(pred) == 1
    assert not np.isnan(pred[0])
    assert pred[0] > 0
