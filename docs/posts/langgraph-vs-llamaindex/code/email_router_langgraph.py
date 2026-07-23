"""
Email router + RAG response drafter — LangGraph version.

Flow (graph traversal over shared state):

    START -> classify --(conditional edge)--> retrieve_billing --+
                       |--> retrieve_support --------------------+--> draft -> END
                       |--> retrieve_sales ----------------------+

Every node reads the shared State dict and returns a partial update.

Run:
    pip install langgraph langchain-openai langchain-core
    export OPENAI_API_KEY=...
    python email_router_langgraph.py
"""

from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1. Model selection — a BaseChatModel instance any node can call
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ---------------------------------------------------------------------------
# 2. One vector store per department
# ---------------------------------------------------------------------------
DEPARTMENT_DOCS = {
    "billing": [
        "Refunds are processed within 5-7 business days to the original "
        "payment method. Annual plans are refundable within 30 days of purchase.",
        "Invoices can be downloaded from Account > Billing > History. "
        "We can re-issue an invoice with a corrected company name or VAT number.",
        "Failed payments are retried automatically after 3 days. Accounts are "
        "suspended after 3 failed attempts but data is retained for 60 days.",
    ],
    "support": [
        "If the app fails to sync, first check Settings > Sync Status. A red "
        "icon means the auth token expired; signing out and back in fixes it.",
        "Error code E-4012 indicates a corrupted local cache. Clearing the "
        "cache from Settings > Storage resolves it without data loss.",
        "We support the last two major versions of Chrome, Firefox, Safari and "
        "Edge. Internet Explorer is not supported.",
    ],
    "sales": [
        "The Team plan is $12/user/month billed annually and includes SSO, "
        "audit logs and priority support. Enterprise adds SLAs and a dedicated CSM.",
        "We offer a 20% discount for registered non-profits and educational "
        "institutions, applied after verification.",
        "Proof-of-concept trials for Enterprise run 30 days and can be extended "
        "once by the account executive.",
    ],
}

STORES = {
    dept: InMemoryVectorStore.from_documents(
        [Document(page_content=t) for t in texts], embedding=embeddings
    )
    for dept, texts in DEPARTMENT_DOCS.items()
}


# ---------------------------------------------------------------------------
# 3. Shared state — the single object that flows through the graph
# ---------------------------------------------------------------------------
class State(TypedDict):
    email: str
    department: str
    context: str
    draft: str


# Structured output schema so classification is typed, not string-parsed
class RouteDecision(BaseModel):
    department: Literal["billing", "support", "sales"] = Field(
        description="The department best suited to answer this email."
    )


# ---------------------------------------------------------------------------
# 4. Nodes — plain functions: State in, partial State update out
# ---------------------------------------------------------------------------
def classify(state: State) -> dict:
    decision = llm.with_structured_output(RouteDecision).invoke(
        "Classify this customer email into one department.\n\n"
        f"EMAIL:\n{state['email']}"
    )
    return {"department": decision.department}


def make_retrieve_node(dept: str):
    """Factory: one retrieval node per department."""

    def retrieve(state: State) -> dict:
        docs = STORES[dept].similarity_search(state["email"], k=2)
        return {"context": "\n\n".join(d.page_content for d in docs)}

    return retrieve


def draft(state: State) -> dict:
    reply = llm.invoke(
        f"You are a {state['department']} representative. Draft a short, "
        "friendly reply to the customer email below. Base your answer ONLY "
        "on the company knowledge provided.\n\n"
        f"COMPANY KNOWLEDGE:\n{state['context']}\n\n"
        f"CUSTOMER EMAIL:\n{state['email']}\n\n"
        "REPLY:"
    )
    return {"draft": reply.content.strip()}


# The routing function used by the conditional edge
def route(state: State) -> str:
    return state["department"]  # "billing" | "support" | "sales"


# ---------------------------------------------------------------------------
# 5. Wire the graph
# ---------------------------------------------------------------------------
builder = StateGraph(State)
builder.add_node("classify", classify)
for dept in DEPARTMENT_DOCS:
    builder.add_node(f"retrieve_{dept}", make_retrieve_node(dept))
builder.add_node("draft", draft)

builder.add_edge(START, "classify")
builder.add_conditional_edges(
    "classify",
    route,
    {dept: f"retrieve_{dept}" for dept in DEPARTMENT_DOCS},
)
for dept in DEPARTMENT_DOCS:
    builder.add_edge(f"retrieve_{dept}", "draft")
builder.add_edge("draft", END)

graph = builder.compile()


# ---------------------------------------------------------------------------
# 6. Run it
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    emails = [
        "Hi, I was charged twice this month and I'd like a refund for the "
        "duplicate charge. Order #4482.",
        "The app keeps showing error E-4012 whenever I open a project. Help!",
        "We're a 40-person non-profit evaluating your Team plan — do you "
        "offer any discounts?",
    ]

    for email in emails:
        result = graph.invoke({"email": email})
        print(f"\n=== routed to: {result['department'].upper()} ===")
        print(result["draft"])
