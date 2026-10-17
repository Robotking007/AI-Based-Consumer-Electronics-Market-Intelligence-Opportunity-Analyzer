"""
sagemaker_pipeline.py - Production SageMaker MLOps Pipeline Definition
Implements Section 4 & 5 of PROJECT.md:
1. Ingest & Preprocessing Step (SageMaker Processing Job)
2. XGBoost Model Training Step (SageMaker Training Job)
3. Model Evaluation Step (SageMaker Processing Job)
4. Quality Gate ConditionStep (RMSE threshold check)
5. Model Registry Step (Champion / Challenger registration in ModelPackageGroup)
"""

import os
import sys
import json
import argparse
from datetime import datetime

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


def create_pipeline_definition(
    bucket_name: str = "consumer-electronics-mlops",
    role_arn: str = "arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole",
    region: str = "ap-south-1",
    rmse_threshold: float = 12.0
) -> dict:
    """
    Construct the native AWS SageMaker Pipeline JSON definition.
    Complies with AWS SageMaker Workflow Schema version 2020-12-01.
    """
    pipeline_name = "ConsumerElectronics-DemandGrowth-Pipeline"
    s3_raw_input = f"s3://{bucket_name}/raw/date={datetime.utcnow().strftime('%Y-%m-%d')}/consumer_electronics_market_20k.csv"
    s3_processed_base = f"s3://{bucket_name}/processed"
    s3_model_base = f"s3://{bucket_name}/models"
    s3_eval_base = f"s3://{bucket_name}/evaluation"

    # 1. Pipeline Parameters
    parameters = [
        {
            "Name": "InputDataUrl",
            "Type": "String",
            "DefaultValue": s3_raw_input
        },
        {
            "Name": "ProcessingInstanceType",
            "Type": "String",
            "DefaultValue": "ml.t3.medium"
        },
        {
            "Name": "TrainingInstanceType",
            "Type": "String",
            "DefaultValue": "ml.m5.xlarge"
        },
        {
            "Name": "RmseThreshold",
            "Type": "Float",
            "DefaultValue": rmse_threshold
        },
        {
            "Name": "ModelApprovalStatus",
            "Type": "String",
            "DefaultValue": "Approved"
        }
    ]

    # AWS Managed Image URIs for ap-south-1 (Mumbai)
    sklearn_image = f"720646828776.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
    xgboost_image = f"720646828776.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:1.5-1"

    # Step 1: Preprocessing & Feature Engineering Job
    step_preprocess = {
        "Name": "PreprocessMarketData",
        "Type": "Processing",
        "Arguments": {
            "ProcessingJobName": "PreprocessMarketData-${SageMaker.ExecutionId}",
            "AppSpecification": {
                "ImageUri": sklearn_image,
                "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/code/preprocessing.py"]
            },
            "RoleArn": role_arn,
            "ProcessingResources": {
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": {"Get": "Parameters.ProcessingInstanceType"},
                    "VolumeSizeInGB": 20
                }
            },
            "ProcessingInputs": [
                {
                    "InputName": "raw-data",
                    "AppManaged": False,
                    "S3Input": {
                        "S3Uri": {"Get": "Parameters.InputDataUrl"},
                        "LocalPath": "/opt/ml/processing/input",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File"
                    }
                },
                {
                    "InputName": "code",
                    "AppManaged": False,
                    "S3Input": {
                        "S3Uri": f"s3://{bucket_name}/code/preprocessing.py",
                        "LocalPath": "/opt/ml/processing/input/code",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File"
                    }
                }
            ],
            "ProcessingOutputConfig": {
                "Outputs": [
                    {
                        "OutputName": "train",
                        "AppManaged": False,
                        "S3Output": {
                            "S3Uri": f"{s3_processed_base}/train",
                            "LocalPath": "/opt/ml/processing/output/train",
                            "S3UploadMode": "EndOfJob"
                        }
                    },
                    {
                        "OutputName": "validation",
                        "AppManaged": False,
                        "S3Output": {
                            "S3Uri": f"{s3_processed_base}/validation",
                            "LocalPath": "/opt/ml/processing/output/validation",
                            "S3UploadMode": "EndOfJob"
                        }
                    },
                    {
                        "OutputName": "test",
                        "AppManaged": False,
                        "S3Output": {
                            "S3Uri": f"{s3_processed_base}/test",
                            "LocalPath": "/opt/ml/processing/output/test",
                            "S3UploadMode": "EndOfJob"
                        }
                    }
                ]
            }
        }
    }

    # Step 2: Training Job (XGBoost Regressor)
    step_train = {
        "Name": "TrainDemandGrowthModel",
        "Type": "Training",
        "DependsOn": ["PreprocessMarketData"],
        "Arguments": {
            "TrainingJobName": "TrainDemandModel-${SageMaker.ExecutionId}",
            "AlgorithmSpecification": {
                "TrainingImage": xgboost_image,
                "TrainingInputMode": "File"
            },
            "RoleArn": role_arn,
            "HyperParameters": {
                "max_depth": "6",
                "eta": "0.08",
                "num_round": "150",
                "subsample": "0.85",
                "colsample_bytree": "0.85",
                "objective": "reg:squarederror",
                "eval_metric": "rmse"
            },
            "InputDataConfig": [
                {
                    "ChannelName": "train",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": f"{s3_processed_base}/train",
                            "S3DataDistributionType": "FullyReplicated"
                        }
                    },
                    "ContentType": "text/csv"
                },
                {
                    "ChannelName": "validation",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": f"{s3_processed_base}/validation",
                            "S3DataDistributionType": "FullyReplicated"
                        }
                    },
                    "ContentType": "text/csv"
                }
            ],
            "OutputDataConfig": {
                "S3OutputPath": s3_model_base
            },
            "ResourceConfig": {
                "InstanceCount": 1,
                "InstanceType": {"Get": "Parameters.TrainingInstanceType"},
                "VolumeSizeInGB": 20
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": 3600
            }
        }
    }

    # Step 3: Model Evaluation Job (computes evaluation.json)
    step_eval = {
        "Name": "EvaluateChallengerModel",
        "Type": "Processing",
        "DependsOn": ["TrainDemandGrowthModel"],
        "Arguments": {
            "ProcessingJobName": "EvaluateChallenger-${SageMaker.ExecutionId}",
            "AppSpecification": {
                "ImageUri": sklearn_image,
                "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/code/evaluate.py"]
            },
            "RoleArn": role_arn,
            "ProcessingResources": {
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": {"Get": "Parameters.ProcessingInstanceType"},
                    "VolumeSizeInGB": 10
                }
            },
            "ProcessingInputs": [
                {
                    "InputName": "model-data",
                    "S3Input": {
                        "S3Uri": f"{s3_model_base}/TrainDemandModel-${{SageMaker.ExecutionId}}/output/model.tar.gz",
                        "LocalPath": "/opt/ml/processing/model",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File"
                    }
                },
                {
                    "InputName": "test-data",
                    "S3Input": {
                        "S3Uri": f"{s3_processed_base}/test",
                        "LocalPath": "/opt/ml/processing/test",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File"
                    }
                },
                {
                    "InputName": "code",
                    "S3Input": {
                        "S3Uri": f"s3://{bucket_name}/code/evaluate.py",
                        "LocalPath": "/opt/ml/processing/input/code",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File"
                    }
                }
            ],
            "ProcessingOutputConfig": {
                "Outputs": [
                    {
                        "OutputName": "evaluation",
                        "S3Output": {
                            "S3Uri": s3_eval_base,
                            "LocalPath": "/opt/ml/processing/evaluation",
                            "S3UploadMode": "EndOfJob"
                        }
                    }
                ]
            }
        },
        "PropertyFiles": [
            {
                "PropertyFileName": "EvaluationReport",
                "OutputName": "evaluation",
                "FilePath": "evaluation.json"
            }
        ]
    }

    # Step 4: Quality Gate ConditionStep (Check if RMSE <= threshold)
    step_condition = {
        "Name": "QualityGate-Condition",
        "Type": "Condition",
        "DependsOn": ["EvaluateChallengerModel"],
        "Arguments": {
            "Conditions": [
                {
                    "Type": "LessThanOrEqualTo",
                    "LeftValue": {
                        "Get": "Steps.EvaluateChallengerModel.OutputProperties.EvaluationReport.regression_metrics.rmse.value"
                    },
                    "RightValue": {"Get": "Parameters.RmseThreshold"}
                }
            ],
            "IfSteps": [
                {
                    "Name": "RegisterApprovedChallengerModel",
                    "Type": "RegisterModel",
                    "Arguments": {
                        "ModelPackageGroupName": "ConsumerElectronicsDemandGrowth",
                        "ModelApprovalStatus": {"Get": "Parameters.ModelApprovalStatus"},
                        "InferenceSpecification": {
                            "Containers": [
                                {
                                    "Image": xgboost_image,
                                    "ModelDataUrl": f"{s3_model_base}/TrainDemandModel-${{SageMaker.ExecutionId}}/output/model.tar.gz"
                                }
                            ],
                            "SupportedContentTypes": ["text/csv"],
                            "SupportedResponseMIMETypes": ["text/csv", "application/json"]
                        }
                    }
                }
            ],
            "ElseSteps": []
        }
    }

    pipeline_dict = {
        "Version": "2020-12-01",
        "Metadata": {
            "PipelineName": pipeline_name,
            "Description": "End-to-End MLOps Pipeline for Consumer Electronics Market Intelligence (Group 10)",
            "LastModified": datetime.utcnow().isoformat()
        },
        "Parameters": parameters,
        "Steps": [step_preprocess, step_train, step_eval, step_condition]
    }

    return pipeline_dict


def save_and_deploy_pipeline(
    bucket_name: str = "consumer-electronics-mlops",
    role_arn: str = None,
    output_json: str = "pipeline_definition.json"
):
    print("Generating AWS SageMaker Pipeline Definition...")
    pipeline_def = create_pipeline_definition(bucket_name=bucket_name)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(pipeline_def, f, indent=2)

    print(f"SageMaker Pipeline Definition saved to: {output_json}")
    print(f"Pipeline Name: {pipeline_def['Metadata']['PipelineName']}")
    print(f"Configured Steps: {[s['Name'] for s in pipeline_def['Steps']]}")

    if BOTO3_AVAILABLE and role_arn:
        try:
            sm_client = boto3.client("sagemaker")
            print(f"\nSubmitting Pipeline definition to AWS SageMaker in ap-south-1...")
            response = sm_client.upsert_pipeline(
                PipelineName=pipeline_def["Metadata"]["PipelineName"],
                PipelineDefinition=json.dumps(pipeline_def),
                PipelineDescription=pipeline_def["Metadata"]["Description"],
                RoleArn=role_arn
            )
            print(f"SageMaker Pipeline successfully upserted on AWS! ARN: {response.get('PipelineArn')}")
        except Exception as e:
            print(f"AWS Submission note: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="consumer-electronics-mlops", help="Target S3 Bucket")
    parser.add_argument("--role-arn", default=None, help="SageMaker Execution Role ARN")
    parser.add_argument("--output", default="pipeline_definition.json", help="Output JSON definition file")

    args = parser.parse_args()
    save_and_deploy_pipeline(
        bucket_name=args.bucket,
        role_arn=args.role_arn,
        output_json=args.output
    )
