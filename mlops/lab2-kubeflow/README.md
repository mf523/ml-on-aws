# Kubeflow

## Environment setup
### Create Role
* Follow this deep link to [create an IAM role with Administrator access](https://console.aws.amazon.com/iam/home#/roles$new?step=review&commonUseCase=EC2%2BEC2&selectedUseCase=EC2&policies=arn:aws:iam::aws:policy%2FAdministratorAccess).
* Confirm that AWS service and EC2 are selected, then click Next: Permissions to view permissions.
* Confirm that AdministratorAccess is checked, then click Next: Tags to assign tags.
* Take the defaults, and click Next: Review to review.
* Enter MLOpsWorkshopEKSRole for the Name, and click Create role.

### Set Cloud9
* Click the grey circle button (in top right corner) and select Manage EC2 Instance.
* Select the instance, then choose Actions / Security / Modify IAM Role.
* Choose MLOpsWorkshopEKSRole from the IAM Role drop down, and select Save.
* Return to your Cloud9 workspace and click the gear icon (in top right corner)
* Select AWS SETTINGS
* Turn off AWS managed temporary credentials
* Close the Preferences tab
Command line
```
rm -vf ${HOME}/.aws/credentials
aws sts get-caller-identity --query Arn | grep MLOpsWorkshopEKSRole -q && echo "IAM role valid" || echo "IAM role NOT valid"
```
Output
```
IAM role valid
```

### Install kubectl
Command line
```
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
kubectl version --client
```
Output
```
Client Version: version.Info{Major:"1", Minor:"20", GitVersion:"v1.20.4", GitCommit:"e87da0bd6e03ec3fea7933c4b5263d151aafd07c", GitTreeState:"clean", BuildDate:"2021-02-18T16:12:00Z", GoVersion:"go1.15.8", Compiler:"gc", Platform:"linux/amd64"}
```

### Install aws-iam-authenticator
Command line
```
curl -o aws-iam-authenticator https://amazon-eks.s3.us-west-2.amazonaws.com/1.19.6/2021-01-05/bin/linux/amd64/aws-iam-authenticator
chmod +x ./aws-iam-authenticator
mkdir -p $HOME/bin && cp ./aws-iam-authenticator $HOME/bin/aws-iam-authenticator && export PATH=$PATH:$HOME/bin
echo 'export PATH=$PATH:$HOME/bin' >> ~/.bashrc
aws-iam-authenticator help
```
Output
```
A tool to authenticate to Kubernetes using AWS IAM credentials

Usage:
  aws-iam-authenticator [command]

...

Use "aws-iam-authenticator [command] --help" for more information about a command.
```

### Install eksctl
Command line
```
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo install -o root -g root -m 0755 /tmp/eksctl /usr/local/bin/eksctl
eksctl version
```
Output
```
0.38.0
```

## Cluster setup
### Create EKS Cluster
Command line
```
eksctl create cluster -f cluster.yaml
```
Output
```
2021-02-18 21:57:25 [ℹ]  eksctl version 0.38.0
2021-02-18 21:57:25 [ℹ]  using region us-west-2
...
2021-02-18 21:57:59 [ℹ]  waiting for CloudFormation stack "eksctl-mlops-workshop-kubeflow-cluster"
...
2021-02-19 00:04:50 [ℹ]  kubectl command should work with "/home/ubuntu/.kube/config", try 'kubectl get nodes'
2021-02-19 00:04:50 [✔]  EKS cluster "mlops-workshop-kubeflow" in "us-west-2" region is ready
```

### Import your EKS Console credentials
Command line
```
c9builder=$(aws cloud9 describe-environment-memberships --environment-id=$C9_PID | jq -r '.memberships[].userArn')
if echo ${c9builder} | grep -q user; then
    ROLEARN=${c9builder}
elif echo ${c9builder} | grep -q assumed-role; then
    assumedrolename=$(echo ${c9builder} | awk -F/ '{print $(NF-1)}')
    ROLEARN=$(aws iam get-role --role-name ${assumedrolename} --query Role.Arn --output text) 
fi
eksctl create iamidentitymapping --cluster mlops-workshop-kubeflow --arn ${ROLEARN} --group system:masters --username admin
```
Oputput
```
2021-02-19 13:23:53 [ℹ]  eksctl version 0.38.0
2021-02-19 13:23:53 [ℹ]  using region us-west-2
2021-02-19 13:23:54 [ℹ]  adding identity "arn:aws:iam::xxxxxxxx:user/xxxx" to auth ConfigMap
```
Command line
```
kubectl describe configmap -n kube-system aws-auth
```
Oputput
```
Name:         aws-auth
Namespace:    kube-system
Labels:       <none>
Annotations:  <none>

Data
====
mapUsers:
----
- groups:
  - system:masters
  userarn: arn:aws:iam::xxxxxxxx:user/xxxx
  username: admin
...
```

### Install kfctl
Command line
```
curl --silent --location "https://github.com/kubeflow/kfctl/releases/download/v1.2.0/kfctl_v1.2.0-0-gbc038f9_linux.tar.gz" | tar xz -C /tmp
sudo install -o root -g root -m 0755 /tmp/kfctl /usr/local/bin/kfctl
kfctl -h
```
Output
```
A client CLI to create kubeflow applications for specific platforms or 'on-prem' 
to an existing k8s cluster.

Usage:
  kfctl [command]

...

Use "kfctl [command] --help" for more information about a command.
```

### Deploy Kubeflow
Command line
```
export CONFIG_URI="https://raw.githubusercontent.com/kubeflow/manifests/v1.2-branch/kfdef/kfctl_aws.v1.2.0.yaml"
#export CONFIG_URI="https://raw.githubusercontent.com/kubeflow/manifests/v1.2-branch/kfdef/kfctl_aws_cognito.v1.2.0.yaml"
export AWS_CLUSTER_NAME=mlops-workshop-kubeflow
mkdir ${AWS_CLUSTER_NAME} && cd ${AWS_CLUSTER_NAME}
wget -O kfctl_aws.yaml $CONFIG_URI
kfctl apply -V -f kfctl_aws.yaml
kubectl -n kubeflow get svc
```
Output
```
...
NAME                                           TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)             AGE
admission-webhook-service                      ClusterIP   10.100.29.26     <none>        443/TCP             2m41s
application-controller-service                 ClusterIP   10.100.40.118    <none>        443/TCP             3m39s
argo-ui                                        NodePort    10.100.158.203   <none>        80:32582/TCP        2m42s
...
pytorch-operator                               ClusterIP   10.100.76.250    <none>        8443/TCP            2m41s
seldon-webhook-service                         ClusterIP   10.100.54.188    <none>        443/TCP             2m41s
tf-job-operator                                ClusterIP   10.100.160.177   <none>        8443/TCP            2m41s
```

### Add user to Kubeflow Dashboard
Command line
```
kubectl edit configmap dex -n auth
```
```
- email: test@kubeflow.org
  hash: $2b$10$ow6fWbPojHUg56hInYmYXe.B7u3frcSR.kuUkQp2EzXs5t0xfMRtS
  username: test
  userID: 08a8684b-db88-4b73-90a9-3cd1661f5466
```
```
kubectl rollout restart deployment dex -n auth
```

### Proxy Kubeflow Dashboard
Commend line
```
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```
* In your Cloud9 environment, click Tools / Preview / Preview Running Application


## References
* https://www.kubeflow.org/docs/aws/
* https://www.eksworkshop.com/
* https://www.getstartedonsagemaker.com/workshop-k8s-pipeline/
* https://github.com/aws-samples/eks-kubeflow-workshop
* https://github.com/data-science-on-aws/workshop
* https://github.com/aws-samples/eks-kubeflow-cloudformation-quick-start

