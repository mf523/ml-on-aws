# sample docker file
#
# Author: Laurence MIAO
# Email: mf523@outlook.com

FROM centos:latest

MAINTAINER Laurence MIAO <mf523@outlook.com>

ARG CONDA_DIR=/opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

#COPY Trainers/requirements_lightgbm.txt /opt/ml/requirements_lightgbm.txt

RUN dnf install -y python3

RUN dnf install -y epel-release
RUN dnf update -y && \
    dnf install -y \
        redhat-rpm-config \
        python3-devel \
        cmake \
        make \
        automake \
        gcc \
        gcc-c++ \
        kernel-devel \
        git \
        wget

RUN dnf install -y \
    java-11-openjdk-devel

ENV JAVA_VERSION=11 \
      JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64" \
      JDK_HOME="/usr/lib/jvm/java-11-openjdk-amd64" \
      JRE_HOME="/usr/lib/jvm/java-11-openjdk-amd64" \
      ANT_VERSION=1.10.3 \
      MAVEN_HOME="/opt/maven" \
      MAVEN_VERSION=3.5.4 \
      MAVEN_CONFIG="/root/.m2" \
      MAVEN_DOWNLOAD_SHA1="22cac91b3557586bb1eba326f2f7727543ff15e3"
  
# Install Maven
RUN set -ex \
      && mkdir -p $MAVEN_HOME \
      && curl -LSso /var/tmp/apache-maven-$MAVEN_VERSION-bin.tar.gz https://apache.org/dist/maven/maven-3/$MAVEN_VERSION/binaries/apache-maven-$MAVEN_VERSION-bin.tar.gz \
      && echo "$MAVEN_DOWNLOAD_SHA1 /var/tmp/apache-maven-$MAVEN_VERSION-bin.tar.gz" | sha1sum -c - \
      && tar xzvf /var/tmp/apache-maven-$MAVEN_VERSION-bin.tar.gz -C $MAVEN_HOME --strip-components=1 \
      && update-alternatives --install /usr/bin/mvn mvn /opt/maven/bin/mvn 10000 \
      && mkdir -p $MAVEN_CONFIG

RUN python3 -m pip install --upgrade pip sagemaker-containers sentencepiece psutil torch torchvision torchtext
RUN python3 -m pip install torchserve
#lightgbm numpy scipy scikit-learn pandas


RUN \
    # python environment
    # wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    # /bin/bash Miniconda3-latest-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    # export PATH="$CONDA_DIR/bin:$PATH" && \
    # conda config --set always_yes yes --set changeps1 no && \
    # lightgbm
    # conda install -q -y numpy scipy "scikit-learn<=0.21.3" pandas && \
    # git clone --recursive --branch stable --depth 1 https://github.com/Microsoft/LightGBM && \
    # cd LightGBM/python-package && python setup.py install && \
    # clean
    dnf remove -y epel-release \
        redhat-rpm-config \
        python3-devel \
        cmake \
        make \
        automake \
        gcc \
        gcc-c++ \
        kernel-devel \
        git \
        wget && \
    dnf clean all && \
    # conda clean -a -y && \
    rm -rf /usr/local/src/*


ADD serve serve

RUN useradd -m model-server \
    && mkdir -p /home/model-server/tmp

COPY ./serve/docker/dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh

RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh \
    && chown -R model-server /home/model-server

COPY ./serve/docker/config.properties /home/model-server/config.properties
RUN mkdir /home/model-server/model-store && chown -R model-server /home/model-server/model-store

EXPOSE 8080 8081

USER model-server
WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp \
      JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64" \
      JDK_HOME="/usr/lib/jvm/java-11-openjdk-amd64" \
      JRE_HOME="/usr/lib/jvm/java-11-openjdk-amd64" \
      ANT_VERSION=1.10.3 \
      MAVEN_HOME="/opt/maven" \
      MAVEN_VERSION=3.5.4 \
      MAVEN_CONFIG="/root/.m2" \
      MAVEN_DOWNLOAD_SHA1="22cac91b3557586bb1eba326f2f7727543ff15e3"

COPY /cifar10 /opt/ml/code
ENV SAGEMAKER_PROGRAM cifar10.py
#ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
#CMD ["serve"]

# inference
ENTRYPOINT ["/usr/bin/python3", "/opt/ml/code/cifar10.py", "--no-save"]
