"""
mock_sagemaker_runner.py - Local SageMaker Pipeline Simulator & Mock Execution Engine
Accurately simulates AWS SageMaker Pipelines locally:
1. Loads pipeline DAG definition complying with SageMaker Workflow Schema 2020-12-01
2. Resolves Pipeline Parameters with CLI overrides
3. Resolves DAG dependencies and executes Steps in topological order:
   - ProcessingStep (PreprocessMarketData) -> mock /opt/ml/processing
   - TrainingStep (TrainDemandGrowthModel) -> mock /opt/ml/input/data & /opt/ml/model
   - ProcessingStep (EvaluateChallengerModel) -> mock evaluation & PropertyFile generation
   - ConditionStep (QualityGate-Condition) -> LeftValue vs RightValue conditional branching
   - RegisterModelStep (RegisterApprovedChallengerModel) -> Mock Model Registry package
4. Tracks execution lifecycle (ExecutionArn, Step status, execution-summary.json)
"""

import os
import sys
import json
import time
import shutil
import tarfile
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

import preprocessing
import train_xgboost
import evaluate


class MockSageMakerPipelineExecutor:
    """
    Simulates AWS SageMaker Pipeline execution engine on the local system.
    """

    def __init__(
        self,
        pipeline_def_path: str = "pipeline_definition.json",
        output_root: str = "mock_sagemaker_runs",
        dataset_path: str = None
    ):
        if not dataset_path:
            dataset_path = "consumer_electronics_market.csv" if os.path.exists("consumer_electronics_market.csv") else "consumer_electronics_market_20k.csv"
        self.pipeline_def_path = pipeline_def_path
        self.output_root = output_root
        self.dataset_path = dataset_path

        with open(pipeline_def_path, "r", encoding="utf-8") as f:
            self.pipeline_def = json.load(f)

        self.pipeline_name = self.pipeline_def.get("Metadata", {}).get("PipelineName", "Mock-SageMaker-Pipeline")
        self.execution_id = f"exec-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{int(time.time()*1000) % 100000}"
        self.execution_arn = f"arn:aws:sagemaker:ap-south-1:533267153786:pipeline/{self.pipeline_name}/execution/{self.execution_id}"
        
        self.run_dir = os.path.join(self.output_root, self.execution_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self.parameters: Dict[str, Any] = {}
        self.step_outputs: Dict[str, Dict[str, str]] = {}
        self.step_properties: Dict[str, Dict[str, Any]] = {}
        self.step_execution_details: List[Dict[str, Any]] = []

    def _init_parameters(self, overrides: Optional[Dict[str, Any]] = None):
        """Parse default parameters from pipeline definition and apply overrides."""
        for param in self.pipeline_def.get("Parameters", []):
            name = param["Name"]
            default_val = param.get("DefaultValue")
            param_type = param.get("Type", "String")

            if overrides and name in overrides:
                val = overrides[name]
            else:
                val = default_val

            # Type conversion
            if param_type == "Float" and val is not None:
                val = float(val)
            elif param_type == "Integer" and val is not None:
                val = int(val)
            elif param_type == "String" and val is not None:
                val = str(val)

            self.parameters[name] = val

    def resolve_expression(self, expr: Any) -> Any:
        """
        Recursively resolves SageMaker Expressions:
        - {"Get": "Parameters.<Name>"}
        - {"Get": "Steps.<StepName>.OutputProperties.<PropertyFile>.<path>"}
        - String interpolation: ${SageMaker.ExecutionId}
        """
        if isinstance(expr, dict):
            if "Std:JsonGet" in expr:
                jg = expr["Std:JsonGet"]
                prop_file_ref = jg.get("PropertyFile")
                if isinstance(prop_file_ref, dict) and "Get" in prop_file_ref:
                    # Format: Steps.EvaluateChallengerModel.PropertyFiles.EvaluationReport
                    parts = prop_file_ref["Get"].split(".")
                    step_name = parts[1]
                    prop_file_name = parts[3]
                else:
                    step_name = jg.get("StepName", "EvaluateChallengerModel")
                    prop_file_name = str(prop_file_ref)

                json_path = jg.get("JsonPath", jg.get("Path", ""))
                sub_keys = json_path.split(".")
                cur = self.step_properties.get(step_name, {}).get(prop_file_name, {})
                for key in sub_keys:
                    if isinstance(cur, dict) and key in cur:
                        cur = cur[key]
                    else:
                        return None
                return cur

            if "Get" in expr:
                path = expr["Get"]
                if path.startswith("Parameters."):
                    param_name = path.split("Parameters.")[1]
                    return self.parameters.get(param_name)
                elif path.startswith("Steps."):
                    # Format: Steps.<StepName>.OutputProperties.<PropName>.<json_path>
                    parts = path.split(".")
                    step_name = parts[1]
                    if parts[2] in ["OutputProperties", "PropertyFiles"]:
                        prop_name = parts[3]
                        json_subpath = parts[4:]
                        cur = self.step_properties.get(step_name, {}).get(prop_name, {})
                        for key in json_subpath:
                            if isinstance(cur, dict) and key in cur:
                                cur = cur[key]
                            else:
                                return None
                        return cur
            return {k: self.resolve_expression(v) for k, v in expr.items()}

        elif isinstance(expr, list):
            return [self.resolve_expression(item) for item in expr]

        elif isinstance(expr, str):
            res = expr.replace("${SageMaker.ExecutionId}", self.execution_id)
            return res

        return expr

    def execute_processing_step(self, step: Dict[str, Any], sample_limit: Optional[int] = None) -> Dict[str, Any]:
        """Simulate a SageMaker Processing Job."""
        step_name = step["Name"]
        step_dir = os.path.join(self.run_dir, f"step_{step_name}")
        os.makedirs(step_dir, exist_ok=True)

        print(f"\n[SageMaker Pipeline Step: {step_name} (Processing)]")
        start_time = time.time()

        args = step.get("Arguments", {})
        app_spec = args.get("AppSpecification", {})
        entrypoint = app_spec.get("ContainerEntrypoint", ["python3"])

        # Determine which script to run based on entrypoint
        script_name = entrypoint[-1] if entrypoint else ""

        if "preprocessing.py" in script_name:
            out_train_dir = os.path.join(step_dir, "output", "train")
            out_val_dir = os.path.join(step_dir, "output", "validation")
            out_test_dir = os.path.join(step_dir, "output", "test")
            output_combined_dir = os.path.join(step_dir, "output")

            os.makedirs(out_train_dir, exist_ok=True)
            os.makedirs(out_val_dir, exist_ok=True)
            os.makedirs(out_test_dir, exist_ok=True)

            print(f" Executing Feature Engineering & Splitting on dataset: {self.dataset_path}")
            train_path, val_path, test_path = preprocessing.run_preprocessing(
                input_csv=self.dataset_path,
                output_dir=output_combined_dir,
                sample_limit=sample_limit
            )

            # Move or copy splits to emulate SageMaker Processing outputs
            shutil.copy2(train_path, os.path.join(out_train_dir, "train.csv"))
            shutil.copy2(val_path, os.path.join(out_val_dir, "validation.csv"))
            shutil.copy2(test_path, os.path.join(out_test_dir, "test.csv"))

            self.step_outputs[step_name] = {
                "train": os.path.join(out_train_dir, "train.csv"),
                "validation": os.path.join(out_val_dir, "validation.csv"),
                "test": os.path.join(out_test_dir, "test.csv"),
                "metadata": os.path.join(output_combined_dir, "feature_metadata.json")
            }

        elif "evaluate.py" in script_name:
            # Locate model artifact and test dataset from inputs
            model_dir = None
            test_path = None

            for proc_input in args.get("ProcessingInputs", []):
                inp_name = proc_input.get("InputName", "")
                if inp_name == "model-data":
                    # Check upstream training step output
                    model_dir = self.step_outputs.get("TrainDemandGrowthModel", {}).get("model_dir")
                elif inp_name == "test-data":
                    test_path = self.step_outputs.get("PreprocessMarketData", {}).get("test")

            if not model_dir or not test_path:
                # Fallback to local default locations if not linked
                model_dir = os.path.join(self.run_dir, "step_TrainDemandGrowthModel", "model")
                test_path = self.step_outputs.get("PreprocessMarketData", {}).get("test", "processed_data/test.csv")

            eval_out_dir = os.path.join(step_dir, "output", "evaluation")
            os.makedirs(eval_out_dir, exist_ok=True)

            print(f" Executing Challenger Model Evaluation against test set: {test_path}")
            eval_report = evaluate.evaluate(
                model_dir=model_dir,
                test_path=test_path,
                output_dir=eval_out_dir
            )

            self.step_outputs[step_name] = {
                "evaluation": os.path.join(eval_out_dir, "evaluation.json")
            }

            # Handle SageMaker PropertyFiles
            for prop in step.get("PropertyFiles", []):
                prop_file_name = prop["PropertyFileName"]
                prop_file_path = os.path.join(eval_out_dir, prop["FilePath"])
                if os.path.exists(prop_file_path):
                    with open(prop_file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if step_name not in self.step_properties:
                        self.step_properties[step_name] = {}
                    self.step_properties[step_name][prop_file_name] = data
                    print(f" Loaded SageMaker PropertyFile '{prop_file_name}' from: {prop_file_path}")

        duration = round(time.time() - start_time, 2)
        print(f" Step '{step_name}' SUCCEEDED in {duration}s")

        return {
            "StepName": step_name,
            "StepType": "Processing",
            "Status": "Succeeded",
            "DurationSeconds": duration,
            "Outputs": self.step_outputs.get(step_name, {})
        }

    def execute_training_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a SageMaker Training Job."""
        step_name = step["Name"]
        step_dir = os.path.join(self.run_dir, f"step_{step_name}")
        model_dir = os.path.join(step_dir, "model")
        os.makedirs(model_dir, exist_ok=True)

        print(f"\n[SageMaker Pipeline Step: {step_name} (Training)]")
        start_time = time.time()

        # Resolve inputs from previous step
        train_path = self.step_outputs.get("PreprocessMarketData", {}).get("train")
        val_path = self.step_outputs.get("PreprocessMarketData", {}).get("validation")

        if not train_path or not os.path.exists(train_path):
            raise FileNotFoundError(f"Training input channel data not found: {train_path}")

        # Resolve hyperparameters
        hyper = step.get("Arguments", {}).get("HyperParameters", {})
        max_depth = int(hyper.get("max_depth", 6))
        eta = float(hyper.get("eta", 0.08))
        n_estimators = int(hyper.get("num_round", 150))
        subsample = float(hyper.get("subsample", 0.85))
        colsample_bytree = float(hyper.get("colsample_bytree", 0.85))

        print(f" Training XGBoost Regressor (max_depth={max_depth}, eta={eta}, n_estimators={n_estimators})...")
        model, val_rmse, val_mae = train_xgboost.train(
            train_path=train_path,
            val_path=val_path,
            output_dir=model_dir,
            max_depth=max_depth,
            eta=eta,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree
        )

        tar_path = os.path.join(model_dir, "model.tar.gz")
        self.step_outputs[step_name] = {
            "model_dir": model_dir,
            "model_tar_gz": tar_path,
            "val_rmse": float(val_rmse),
            "val_mae": float(val_mae)
        }

        duration = round(time.time() - start_time, 2)
        print(f" Step '{step_name}' SUCCEEDED in {duration}s -> Artifact: {tar_path}")

        return {
            "StepName": step_name,
            "StepType": "Training",
            "Status": "Succeeded",
            "DurationSeconds": duration,
            "Outputs": {
                "ModelArtifact": tar_path,
                "ValidationRMSE": float(val_rmse),
                "ValidationMAE": float(val_mae)
            }
        }

    def execute_condition_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a SageMaker ConditionStep (Quality Gate)."""
        step_name = step["Name"]
        print(f"\n[SageMaker Pipeline Step: {step_name} (Condition / Quality Gate)]")
        start_time = time.time()

        args = step.get("Arguments", {})
        conditions = args.get("Conditions", [])
        overall_condition_met = True
        condition_results = []

        for cond in conditions:
            cond_type = cond.get("Type")
            left_val = self.resolve_expression(cond.get("LeftValue"))
            right_val = self.resolve_expression(cond.get("RightValue"))

            passed = False
            if cond_type == "LessThanOrEqualTo":
                passed = float(left_val) <= float(right_val)
            elif cond_type == "LessThan":
                passed = float(left_val) < float(right_val)
            elif cond_type == "GreaterThanOrEqualTo":
                passed = float(left_val) >= float(right_val)
            elif cond_type == "GreaterThan":
                passed = float(left_val) > float(right_val)
            elif cond_type == "Equals":
                passed = left_val == right_val

            print(f" Condition: {cond_type} | Challenger RMSE: {left_val} vs Quality Gate Threshold: {right_val} -> {'PASS' if passed else 'FAIL'}")
            condition_results.append({
                "Type": cond_type,
                "LeftValue": left_val,
                "RightValue": right_val,
                "Passed": passed
            })

            if not passed:
                overall_condition_met = False

        executed_steps = []
        if overall_condition_met:
            print(" [QUALITY GATE PASSED] Challenger model approved! Proceeding to IfSteps (Model Registration)...")
            for sub_step in args.get("IfSteps", []):
                sub_res = self.execute_step(sub_step)
                executed_steps.append(sub_res)
        else:
            print(" [QUALITY GATE FAILED] Challenger model rejected! Quality threshold breached. Running ElseSteps...")
            for sub_step in args.get("ElseSteps", []):
                sub_res = self.execute_step(sub_step)
                executed_steps.append(sub_res)

        duration = round(time.time() - start_time, 2)
        return {
            "StepName": step_name,
            "StepType": "Condition",
            "Status": "Succeeded",
            "DurationSeconds": duration,
            "ConditionMet": overall_condition_met,
            "ConditionDetails": condition_results,
            "ExecutedSubSteps": executed_steps
        }

    def execute_register_model_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate SageMaker Model Registry packaging."""
        step_name = step["Name"]
        print(f"\n[SageMaker Pipeline Step: {step_name} (RegisterModel)]")
        start_time = time.time()

        args = step.get("Arguments", {})
        group_name = args.get("ModelPackageGroupName", "ConsumerElectronicsDemandGrowth")
        approval_status = self.resolve_expression(args.get("ModelApprovalStatus", "Approved"))

        # Fetch metrics from evaluation
        eval_metrics = self.step_properties.get("EvaluateChallengerModel", {}).get("EvaluationReport", {}).get("regression_metrics", {})
        rmse_val = eval_metrics.get("rmse", {}).get("value")
        mae_val = eval_metrics.get("mae", {}).get("value")
        r2_val = eval_metrics.get("r2", {}).get("value")

        model_tar = self.step_outputs.get("TrainDemandGrowthModel", {}).get("model_tar_gz")

        registry_entry = {
            "ModelPackageArn": f"arn:aws:sagemaker:ap-south-1:533267153786:model-package/{group_name}/1",
            "ModelPackageGroupName": group_name,
            "ModelPackageVersion": 1,
            "ModelApprovalStatus": approval_status,
            "CreationTime": datetime.utcnow().isoformat(),
            "ModelArtifact": model_tar,
            "Metrics": {
                "rmse": rmse_val,
                "mae": mae_val,
                "r2": r2_val
            }
        }

        registry_file = os.path.join(self.run_dir, "registered_model_package.json")
        with open(registry_file, "w", encoding="utf-8") as f:
            json.dump(registry_entry, f, indent=2)

        print(f" Model Package registered in Group '{group_name}' with Status: '{approval_status}'")
        print(f" Registry Metadata saved to: {registry_file}")

        duration = round(time.time() - start_time, 2)
        return {
            "StepName": step_name,
            "StepType": "RegisterModel",
            "Status": "Succeeded",
            "DurationSeconds": duration,
            "RegisteredModel": registry_entry
        }

    def execute_step(self, step: Dict[str, Any], sample_limit: Optional[int] = None) -> Dict[str, Any]:
        """Dispatch step to appropriate executor based on step Type."""
        step_type = step.get("Type")
        if step_type == "Processing":
            res = self.execute_processing_step(step, sample_limit=sample_limit)
        elif step_type == "Training":
            res = self.execute_training_step(step)
        elif step_type == "Condition":
            res = self.execute_condition_step(step)
        elif step_type == "RegisterModel":
            res = self.execute_register_model_step(step)
        else:
            raise NotImplementedError(f"Unsupported SageMaker Step Type: {step_type}")

        self.step_execution_details.append(res)
        return res

    def run(
        self,
        parameter_overrides: Optional[Dict[str, Any]] = None,
        sample_limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute full SageMaker Pipeline locally."""
        print("="*75)
        print(f" AWS SAGEMAKER PIPELINE LOCAL SIMULATOR & MOCK RUNNER")
        print(f" Pipeline Name : {self.pipeline_name}")
        print(f" Execution ARN : {self.execution_arn}")
        print(f" Run Directory : {os.path.abspath(self.run_dir)}")
        print("="*75)

        self._init_parameters(overrides=parameter_overrides)
        print("\nResolved Pipeline Parameters:")
        for k, v in self.parameters.items():
            print(f" - {k}: {v}")

        start_time = time.time()
        pipeline_status = "Succeeded"

        try:
            steps = self.pipeline_def.get("Steps", [])
            for step in steps:
                self.execute_step(step, sample_limit=sample_limit)
        except Exception as e:
            pipeline_status = "Failed"
            print(f"\n[FATAL ERROR] SageMaker Pipeline execution failed: {e}")
            raise e
        finally:
            total_duration = round(time.time() - start_time, 2)
            summary = {
                "PipelineExecutionArn": self.execution_arn,
                "PipelineExecutionDisplayName": self.execution_id,
                "PipelineExecutionStatus": pipeline_status,
                "CreationTime": datetime.utcnow().isoformat(),
                "TotalDurationSeconds": total_duration,
                "Parameters": self.parameters,
                "StepExecutionDetails": self.step_execution_details,
                "StepOutputs": self.step_outputs,
                "Properties": self.step_properties
            }

            summary_file = os.path.join(self.run_dir, "execution_summary.json")
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)

            # Also maintain a latest pointer
            latest_file = os.path.join(self.output_root, "latest_execution.json")
            with open(latest_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)

            print("\n" + "="*75)
            print(f" SAGEMAKER PIPELINE EXECUTION COMPLETED: {pipeline_status.upper()}")
            print(f" Total Duration : {total_duration}s")
            print(f" Trace Summary  : {summary_file}")
            print("="*75 + "\n")

        return summary


def run_mock_pipeline(
    pipeline_def_path: str = "pipeline_definition.json",
    rmse_threshold: Optional[float] = None,
    sample_limit: Optional[int] = None,
    dataset_path: str = None
) -> Dict[str, Any]:
    """Convenience helper to initialize and run the local mock pipeline."""
    if not dataset_path:
        dataset_path = "consumer_electronics_market.csv" if os.path.exists("consumer_electronics_market.csv") else "consumer_electronics_market_20k.csv"
    executor = MockSageMakerPipelineExecutor(
        pipeline_def_path=pipeline_def_path,
        dataset_path=dataset_path
    )
    overrides = {}
    if rmse_threshold is not None:
        overrides["RmseThreshold"] = rmse_threshold

    return executor.run(parameter_overrides=overrides, sample_limit=sample_limit)


if __name__ == "__main__":
    default_data = "consumer_electronics_market.csv" if os.path.exists("consumer_electronics_market.csv") else "consumer_electronics_market_20k.csv"
    parser = argparse.ArgumentParser(description="Run Mock SageMaker Pipeline locally")
    parser.add_argument("--pipeline-def", default="pipeline_definition.json", help="Path to pipeline definition JSON")
    parser.add_argument("--threshold", type=float, default=None, help="Override Quality Gate RMSE Threshold")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit dataset rows for fast testing")
    parser.add_argument("--force-fail", action="store_true", help="Force Quality Gate to fail (tests condition rejection)")
    parser.add_argument("--data", default=default_data, help="Input dataset CSV")

    args = parser.parse_args()
    threshold = 0.5 if args.force_fail else args.threshold

    run_mock_pipeline(
        pipeline_def_path=args.pipeline_def,
        rmse_threshold=threshold,
        sample_limit=args.sample_limit,
        dataset_path=args.data
    )
