interface ImportMetaEnv {
  readonly VITE_LANGGRAPH_API_URL: string;
  readonly VITE_LANGGRAPH_ASSISTANT_ID: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
