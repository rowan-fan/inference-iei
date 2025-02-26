FROM xprobe/xinference:v1.3.0.post2

RUN pip install --upgrade -i https://pypi.org/simple pip && \
    pip install  "sglang[all]>=0.4.3.post2" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer && \
    pip install vllm==0.7.3 && \
    pip install transformers==4.48.3 && \
    rm -rf /root/.cache/pip/* /tmp/* /var/tmp/*