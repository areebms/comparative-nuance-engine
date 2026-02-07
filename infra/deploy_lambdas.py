#!/usr/bin/env python3
import os

import boto3
from botocore.exceptions import ClientError


AWS_URI_PREFIX = os.environ.get("AWS_URI_PREFIX")
AWS_ECR_REPO = os.environ.get("AWS_ECR_REPO")
LAMBDA_ROLE_ARN = os.environ.get("LAMBDA_ROLE_ARN")

MEMORY = 1024
TIMEOUT = 300

services = {
    "lambda-scrape": f"{AWS_URI_PREFIX}/{AWS_ECR_REPO}/lambda-scrape:0.1",
    "lambda-tokenize": f"{AWS_URI_PREFIX}/{AWS_ECR_REPO}/lambda-tokenize:0.1",
    "lambda-generate-kvector": f"{AWS_URI_PREFIX}/{AWS_ECR_REPO}/lambda-generate-kvector:0.1",
}

def lambda_exists(lambda_client, function_name: str) -> bool:
    try:
        lambda_client.get_function(FunctionName=function_name)
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
            return False
        raise

if __name__ == "__main__":
    region = os.environ.get("AWS_REGION")

    lambda_client = boto3.client("lambda", region_name=region)

    for service_name, image_uri in services.items():
        function_name =  '-'.join(service_name.split('-')[1:])

        if not lambda_exists(lambda_client, function_name):
            print(f"[CREATE] {function_name} -> {image_uri}")
            lambda_client.create_function(
                FunctionName=function_name,
                PackageType="Image",
                Code={"ImageUri": image_uri},
                Role=LAMBDA_ROLE_ARN,
                MemorySize=MEMORY,
                Timeout=TIMEOUT,
            )
        else: # TODO: Update only if different.
            print(f"[UPDATE] {function_name} -> {image_uri}")
            lambda_client.update_function_code(FunctionName=function_name, ImageUri=image_uri)
            lambda_client.update_function_configuration(
                FunctionName=function_name, MemorySize=MEMORY, Timeout=TIMEOUT
            )

        # prevent racey follow-up calls
        lambda_client.get_waiter("function_active_v2").wait(FunctionName=function_name)

