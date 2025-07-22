#! /bin/bash
cd $(dirname $0)

docker build -t ichat:test -f Dockerfile.ichat .