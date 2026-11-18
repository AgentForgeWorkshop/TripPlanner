import os
from langchain.agents import create_agent
from langchain.tools import tool
from conference_agent import conference_agent


@tool
def find_conferences(query: str) -> str:
    """Delegate conference discovery to the conference expert."""
    result = conference_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content


orchestrator = create_agent(
    model=os.environ["MODEL"],
    tools=[find_conferences],
    system_prompt="""
    You coordinate specialists to help users plan
    conference-related travel. Delegate conference discovery
    to the conference expert. Report missing information
    rather than inventing it.
    """
)


if __name__ == "__main__":
    result = orchestrator.invoke({
        "messages": [{
            "role": "user",
            "content": "Help me attend an ecology conference in March 2027."
        }]
    })

    print(result["messages"][-1].content)