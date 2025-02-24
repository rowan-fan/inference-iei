#! /bin/bash
cd $(dirname $0)/../../..

docker build -t xinference-iei:v1.2.2-0.7.2-0.4.2 -f xinference/deploy/docker/legacy.Dockerfile .