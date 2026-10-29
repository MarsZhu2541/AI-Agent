# AI Agent
A stream chat ai agent web application build by react and python
## Demo
http://139.196.93.191/assistantui


# Python langgraph fast api

online search tool: https://www.tavily.com/

suggested Open AI api: https://api.chatanywhere.tech/

cd backend and creat .env file to setup your api keys

```
python3.12 -m venv .venv

source .venv/bin/activate

pip install -e . "langgraph-cli[inmem]"

uvicorn agent.api:app --reload --port 8000 --env-file .env
```

# React + Vite
https://www.assistant-ui.com/
```
npm install
npm run dev
npm run build
```