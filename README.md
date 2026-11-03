# AI Agent
A stream chat ai agent web application build by react and python
## Demo
http://139.196.93.191/assistantui


# Python langgraph fast api

online search tool: https://www.tavily.com/

suggested Open AI api: https://api.chatanywhere.tech/

cd backend and creat .env file to setup your api keys

LOL RAG 

```

/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate
.venv/bin/python -m pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
uvicorn agent.api:app --reload --port 8000 --env-file .env

```

# React + Vite
https://www.assistant-ui.com/
```
npm install
npm run dev
npm run build
```

# PostgreSQL

Chat history and lore embeddings are stored in PostgreSQL with pgvector.
Set `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`,
and `POSTGRES_DATABASE` in `backend/.env`.

```
docker run -d \
  --name pgvector-db \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_DB=ai_agent \
  -p 5432:5432 \
  docker.1ms.run/pgvector/pgvector:pg16

docker exec -it pgvector-db psql -U postgres -d ai_agent -c "CREATE EXTENSION IF NOT EXISTS vector;"
```