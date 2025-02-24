FROM xprobe/xinference:v1.2.2

RUN pip install --upgrade -i https://pypi.org/simple pip && \
    pip install vllm==0.7.2 && \
    pip install "sglang[all]>=0.4.2.post4" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer && \
    rm -rf /root/.cache/pip/* /tmp/* /var/tmp/*