FROM xprobe/xinference:v1.6.0.post1

ADD ./xinference /usr/local/lib/python3.10/dist-packages/xinference

RUN pip install --upgrade pip \
    && pip install -U "magic-pdf[full]" -i https://mirrors.aliyun.com/pypi/simple \
    && pip uninstall -y magic-pdf \
    && rm -rf /root/.cache/pip

WORKDIR /workspace
ADD docker-iei/run.py /workspace/run.py
ADD docker-iei/start.sh /workspace/start.sh
ADD docker-iei/sensitive_server /workspace/sensitive_server