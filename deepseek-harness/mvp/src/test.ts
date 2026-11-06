import { createComplete1 } from "./client.js";



async function main() {
    const complete = createComplete1("https://api.chatanywhere.tech/v1", "sk-JJzHQJa2n6zBeBb84GryzQU58hGHrUgZKdUuAHFCWUlbpzOR", "deepseek-v4.1-flash");
    const answer = await complete([{ role: "user", content: "你好" }], []);
    console.log(answer);
}

main().catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`${message}\n`);
    process.exit(1);
});
