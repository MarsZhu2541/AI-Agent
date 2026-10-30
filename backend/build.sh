docker build --platform linux/amd64 -t marszhu2541/langgraph:0.1 .
docker save marszhu2541/langgraph:0.1 >langgraph_0.1.tar
scp langgraph_0.1.tar Mars@139.196.93.191:/home/Mars/langgraph_0.1.tar
# docker run --name langgraph -p 8000:8000 -d \
#   --env-file /home/Mars/.env \
#   marszhu2541/langgraph:0.1