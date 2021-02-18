# Kubeflow

## Environment setup
### Install kuberctl
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
eksctl -h
```
Output
```
The official CLI for Amazon EKS

Usage: eksctl [command] [flags]

...

Use 'eksctl [command] --help' for more information about a command.
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
2021-02-18 21:57:59 [ℹ]  waiting for CloudFormation stack "eksctl-kfworkshop-cluster"
...

```

### Install Kubeflow
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
