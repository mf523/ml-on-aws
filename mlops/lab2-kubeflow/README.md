# Kubeflow

## Install kuberctl
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

## Install aws-iam-authenticator
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

Available Commands:
  help        Help about any command
  init        Pre-generate certificate, private key, and kubeconfig files for the server.
  server      Run a webhook validation server suitable that validates tokens using AWS IAM
  token       Authenticate using AWS IAM and get token for Kubernetes
  verify      Verify a token for debugging purpose
  version     Version will output the current build information

Flags:
  -i, --cluster-id ID                 Specify the cluster ID, a unique-per-cluster identifier for your aws-iam-authenticator installation.
  -c, --config filename               Load configuration from filename
      --feature-gates mapStringBool   A set of key=value pairs that describe feature gates for alpha/experimental features. Options are:
                                      AllAlpha=true|false (ALPHA - default=false)
                                      IAMIdentityMappingCRD=true|false (ALPHA - default=false)
  -h, --help                          help for aws-iam-authenticator
  -l, --log-format string             Specify log format to use when logging to stderr [text or json] (default "text")

Use "aws-iam-authenticator [command] --help" for more information about a command.
```

## Install eksctl
Command line
```
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin
eksctl -h
```
Output
```
The official CLI for Amazon EKS

Usage: eksctl [command] [flags]

Commands:
  eksctl associate                       Associate resources with a cluster
  eksctl completion                      Generates shell completion scripts for bash, zsh or fish
  eksctl create                          Create resource(s)
  eksctl delete                          Delete resource(s)
  eksctl disassociate                    Disassociate resources from a cluster
  eksctl drain                           Drain resource(s)
  eksctl enable                          Enable features in a cluster
  eksctl generate                        Generate gitops manifests
  eksctl get                             Get resource(s)
  eksctl help                            Help about any command
  eksctl scale                           Scale resources(s)
  eksctl set                             Set values
  eksctl unset                           Unset values
  eksctl update                          Update resource(s)
  eksctl upgrade                         Upgrade resource(s)
  eksctl utils                           Various utils
  eksctl version                         Output the version of eksctl

Common flags:
  -C, --color string   toggle colorized logs (valid options: true, false, fabulous) (default "true")
  -h, --help           help for this command
  -v, --verbose int    set log level, use 0 to silence, 4 for debugging and 5 for debugging with AWS debug logging (default 3)

Use 'eksctl [command] --help' for more information about a command.
```
