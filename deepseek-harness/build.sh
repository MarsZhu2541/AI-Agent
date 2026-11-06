#!/bin/sh
set -eu
cd "$(dirname "$0")"
docker build --platform linux/amd64 -t deepseek-harness:0.2.1 .
docker save deepseek-harness:0.2.1 > deepseek-harness_0.2.1.tar
scp deepseek-harness_0.2.1.tar Mars@139.196.93.191:/home/Mars/deepseek-harness_0.2.1.tar



docker run -d --name deepseek-harness --restart unless-stopped \
  -p 3080:3080 \
  -v deepseek-harness-data:/data/dsh \
  -v deepseek-harness-workspace:/workspace \
  deepseek-harness:0.2.1
