import asyncio
from browser_use import Agent
from langchain_ollama import ChatOllama

async def main():
    # Point to your local Ollama server & model
    llm = ChatOllama(
        base_url="http://localhost:11434",
        model="llama3.1:8b",
        temperature=0.2,
        num_ctx=8192,  # optional: bigger context if your RAM allows
    )

    # Give the agent a concrete browsing task
    task = (
        "Open https://news.ycombinator.com, click the 'Show' link, "
        "extract the title and URL of the first post, and return them as JSON."
    )

    agent = Agent(
        task=task,
        llm=llm,
        headless=False,          # set True to run without a visible browser
        max_actions=12,          # safety stop
    )

    result = await agent.run()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
