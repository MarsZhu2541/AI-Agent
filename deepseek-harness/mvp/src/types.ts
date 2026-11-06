export type ToolCall = {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
};

export type Message =
  | { role: "system" | "user"; content: string }
  | { role: "assistant"; content: string | null; tool_calls?: ToolCall[] }
  | { role: "tool"; tool_call_id: string; content: string };

export type ToolContext = {
  workspace: string;
};

export type Tool = {
  name: string;
  description: string;
  parameters: {
    type: "object";
    properties: Record<string, unknown>;
    required?: string[];
    additionalProperties: false;
  };
  execute: (args: unknown, context: ToolContext) => Promise<string>;
};

export type ToolSchema = {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Tool["parameters"];
  };
};

export type Completer = (messages: Message[], tools: ToolSchema[]) => Promise<Extract<Message, { role: "assistant" }>>;
