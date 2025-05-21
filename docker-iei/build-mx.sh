#! /bin/bash
cd $(dirname $0)

docker build -t inais/llm-serving:8.0-mx -f Dockerfile.mx ..
