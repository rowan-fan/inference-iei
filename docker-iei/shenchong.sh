rm -r epaipdfparser
git clone http://100.60.184.63/shenchong/epaipdfparser.git

docker build -t custom:v1.0 -f Dockerfile.tmp ..
docker tag custom:v1.0 100.60.131.23:5000/inais/llm-serving:8.0
docker push 100.60.131.23:5000/inais/llm-serving:8.0
echo 'end'
