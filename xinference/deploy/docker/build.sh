#! /bin/bash
cd $(dirname $0)/../../..

docker build -t xinference-iei:v1.3.0.post2-0.7.3-0.4.3 -f xinference/deploy/docker/legacy.Dockerfile .