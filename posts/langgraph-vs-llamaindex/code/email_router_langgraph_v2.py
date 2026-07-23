"""
Email router + RAG drafter with a 3-reply cap — LangGraph version.

Business rule: the agent may reply to a customer thread at most 3 times.
On the 4th inbound email, it stops drafting and forwards the whole thread
to the department inbox for human review.

Graph:

    START -> gate --(reply_count >= 3)--> escalate -> END
              |
              +--(else)--> classify --> retrieve_<dept> --> draft -> END

Per-customer threads persist across invocations via a checkpointer +
thread_id: each graph.invoke() only sends the NEW inbound email; the
checkpointer restores messages, department, and reply_count.

Run:
    pip install langgraph langchain-openai langchain-core
    export OPENAI_API_KEY=...
    python email_router_langgraph_v2.py
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

MAX_AGENT_REPLIES = 3

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ---------------------------------------------------------------------------
# Department knowledge bases
# ---------------------------------------------------------------------------
DEPARTMENT_DOCS = {
    "billing": [
        "Refunds are processed within 5-7 business days to the original "
        "payment method. Annual plans are refundable within 30 days of purchase.",
        "Invoices can be downloaded from Account > Billing > History.",
        "Failed payments are retried automatically after 3 days. Accounts are "
        "suspended after 3 failed attempts but data is retained for 60 days.",
    ],
    "support": [
        "If the app fails to sync, check Settings > Sync Status. A red icon "
        "means the auth token expired; signing out and back in fixes it.",
        "Error code E-4012 indicates a corrupted local cache. Clearing the "
        "cache from Settings > Storage resolves it without data loss.",
    ],
    "sales": [
        "The Team plan is $12/user/month billed annually and includes SSO, "
        "audit logs and priority support.",
        "We offer a 20% discount for registered non-profits and educational "
        "institutions, applied after verification.",
    ],
}

STORES = {
    dept: InMemoryVectorStore.from_documents(
        [Document(page_content=t) for t in texts], embedding=embeddings
    )
    for dept, texts in DEPARTMENT_DOCS.items()
}


# ---------------------------------------------------------------------------
# Shared state — persisted per thread_id by the checkpointer
# ---------------------------------------------------------------------------
class State(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]  # full thread
    department: str
    context: str
    reply_count: int
    escalated: bool
    forward: str  # the internal forward composed on escalation


class RouteDecision(BaseModel):
    department: Literal["billing", "support", "sales"] = Field(
        description="The department best suited to answer this email."
    )


def _latest_customer_msg(state: State) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _transcript(state: State) -> str:
    lines = []
    for m in state["messages"]:
        who = "CUSTOMER" if isinstance(m, HumanMessage) else "AGENT"
        lines.append(f"{who}: {m.content}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def gate(state: State) -> dict:
    """Anchor node for the reply-cap branch. No state change."""
    return {}


def route_gate(state: State) -> str:
    if state.get("reply_count", 0) >= MAX_AGENT_REPLIES:
        return "escalate"
    return "classify"


def classify(state: State) -> dict:
    if state.get("department"):  # already routed on a previous turn
        return {}
    decision = llm.with_structured_output(RouteDecision).invoke(
        "Classify this customer email into one department.\n\n"
        f"EMAIL:\n{_latest_customer_msg(state)}"
    )
    return {"department": decision.department}


def route_department(state: State) -> str:
    return state["department"]


def make_retrieve_node(dept: str):
    def retrieve(state: State) -> dict:
        docs = STORES[dept].similarity_search(_latest_customer_msg(state), k=2)
        return {"context": "\n\n".join(d.page_content for d in docs)}

    return retrieve


def draft(state: State) -> dict:
    reply = llm.invoke(
        f"You are a {state['department']} representative. Continue this "
        "email thread with a short, friendly reply to the customer's latest "
        "message. Base your answer ONLY on the company knowledge provided.\n\n"
        f"COMPANY KNOWLEDGE:\n{state['context']}\n\n"
        f"THREAD SO FAR:\n{_transcript(state)}\n\n"
        "REPLY:"
    )
    return {
        "messages": [AIMessage(content=reply.content.strip())],
        "reply_count": state.get("reply_count", 0) + 1,
    }


def escalate(state: State) -> dict:
    dept = state.get("department", "support")
    forward = (
        f"To: {dept}@company.com\n"
        f"Subject: [ESCALATION] Thread exceeded {MAX_AGENT_REPLIES} automated "
        "replies — human review required\n\n"
        "The automated agent has reached its reply limit on this thread. "
        "Full transcript below.\n\n"
        f"{_transcript(state)}"
    )
    courtesy = AIMessage(
        content=(
            "Thanks for your patience — I've forwarded your thread to our "
            f"{dept} team, and a colleague will follow up with you directly."
        )
    )
    return {"escalated": True, "forward": forward, "messages": [courtesy]}


# ---------------------------------------------------------------------------
# Wire the graph
# ---------------------------------------------------------------------------
builder = StateGraph(State)
builder.add_node("gate", gate)
builder.add_node("classify", classify)
for dept in DEPARTMENT_DOCS:
    builder.add_node(f"retrieve_{dept}", make_retrieve_node(dept))
builder.add_node("draft", draft)
builder.add_node("escalate", escalate)

builder.add_edge(START, "gate")
builder.add_conditional_edges(
    "gate", route_gate, {"escalate": "escalate", "classify": "classify"}
)
builder.add_conditional_edges(
    "classify",
    route_department,
    {dept: f"retrieve_{dept}" for dept in DEPARTMENT_DOCS},
)
for dept in DEPARTMENT_DOCS:
    builder.add_edge(f"retrieve_{dept}", "draft")
builder.add_edge("draft", END)
builder.add_edge("escalate", END)

graph = builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Simulate one customer thread: 4 inbound emails -> 3 replies + 1 escalation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    thread = {"configurable": {"thread_id": "customer-4482"}}

    inbound = [
        "Hi, I was charged twice this month and I'd like a refund for the "
        "duplicate charge. Order #4482.",
        "Thanks — how long will the refund take to show up on my card?",
        "It's been 2 days and I don't see it yet. Can you check?",
        "Still nothing. This is really frustrating, I need this resolved now.",
    ]

    for i, email in enumerate(inbound, 1):
        result = graph.invoke({"messages": [HumanMessage(content=email)]}, thread)
        print(f"\n===== inbound #{i} =====")
        print(f"CUSTOMER: {email}")
        if result.get("escalated"):
            print(f"AGENT (auto-reply): {result['messages'][-1].content}")
            print("\n--- INTERNAL FORWARD ---")
            print(result["forward"])
        else:
            print(f"AGENT ({result['department']}): "
                  f"{result['messages'][-1].content}")
