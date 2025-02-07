FROM vllm/vllm-openai:v0.7.2

COPY . /opt/inference
WORKDIR /opt/inference

ENV NVM_DIR /usr/local/nvm
ENV NODE_VERSION 14.21.1

RUN apt-get -y update \
  && apt install -y curl procps git libgl1 ffmpeg \
  # 添加 python 软链接
  && ln -s /usr/bin/python3 /usr/bin/python \
  # upgrade libstdc++ and libc for llama-cpp-python
  && printf "\ndeb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse" >> /etc/apt/sources.list \
  && apt-get -y update \
  && apt-get install -y --only-upgrade libstdc++6 && apt install -y libc6 \
  && mkdir -p $NVM_DIR \
  && curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash \
  && . $NVM_DIR/nvm.sh \
  && nvm install $NODE_VERSION \
  && nvm alias default $NODE_VERSION \
  && nvm use default \
  && apt-get -yq clean

ENV PATH $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib
# ref: https://github.com/AutoGPTQ/AutoGPTQ/issues/730
ENV PYPI_RELEASE="1"

ARG PIP_INDEX=https://pypi.org/simple
RUN pip install --upgrade -i "$PIP_INDEX" pip && \
    pip install -i "$PIP_INDEX" "diskcache>=5.6.1" "jinja2>=2.11.3"  && \
    # use pre-built whl package for llama-cpp-python, otherwise may core dump when init llama in some envs
    # pip install "llama-cpp-python>=0.2.82" -i https://abetlen.github.io/llama-cpp-python/whl/cu124 && \
    pip install -i "$PIP_INDEX" --upgrade-strategy only-if-needed -r /opt/inference/xinference/deploy/docker/requirements_core.txt && \
    # pip install -i "$PIP_INDEX" --no-deps sglang && \
    pip uninstall flashinfer -y && \
    # TODO: 需要指定版本，下载太慢了 0.1.6+cu124torch2.4 或 0.2.0+cu124torch2.4
    pip install flashinfer==0.1.6+cu124torch2.4 -i https://flashinfer.ai/whl/cu124/torch2.4 && \
    cd /opt/inference && \
    python3 setup.py build_web && \
    git restore . && \
    pip install -i "$PIP_INDEX" --no-deps "." && \
    # clean packages
    pip cache purge

# # Overwrite the entrypoint of vllm's base image
ENTRYPOINT []
CMD ["/bin/bash"]
