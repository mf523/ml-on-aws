#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
image=$1
docker_file=$2

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name> [docker-file]"
    exit 1
fi

if [ "$docker_file" == "" ]
then
    docker_file="Dockerfile"
fi

# chmod +x decision_trees/train
# chmod +x decision_trees/serve

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi


# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-us-west-2}

if [ ${region} == "cn-north-1" ]; then
    # ecr.cn-north-1.amazonaws.com.cn
    fullname="${account}.dkr.ecr.${region}.amazonaws.com.cn/${image}:latest"
elif [ ${region} == "cn-northwest-1" ]; then
    # ecr.cn-northwest-1.amazonaws.com.cn
    fullname="${account}.dkr.ecr.${region}.amazonaws.com.cn/${image}:latest"
else
    fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"
fi


# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker build -t ${image} -f ${docker_file} .
docker tag ${image} ${fullname}

docker push ${fullname}
