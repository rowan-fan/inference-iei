#! /bin/bash
cd $(dirname $0)/../../..

docker build -t xinference-iei:v1.2.1-0.7.1 -f xinference/deploy/docker/core.Dockerfile .