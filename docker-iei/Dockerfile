FROM xinference-iei:v1.3.0.post2-0.7.3-0.4.3-cpu

ADD ./xinference /usr/local/lib/python3.10/dist-packages/xinference

WORKDIR /workspace
ADD docker-iei/run.py /workspace/run.py
ADD docker-iei/start.sh /workspace/start.sh
ADD docker-iei/sensitive_server /workspace/sensitive_server
