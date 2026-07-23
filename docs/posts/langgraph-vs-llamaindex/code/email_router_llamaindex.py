"""
Email router + RAG response drafter — LlamaIndex Workflows version.

Flow (event-driven):
    StartEvent(email)
        -> classify   (LLM picks a department)   -> RoutedEvent
        -> retrieve   (RAG over that dept's index) -> ContextReadyEvent
        -> draft      (LLM writes the reply)     -> StopEvent(result)

Run:
    pip install llama-index-core llama-index-llms-openai llama-index-embeddings-openai
    export OPENAI_API_KEY=...
    python email_router_llamaindex.py
"""

import asyncio

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# ---------------------------------------------------------------------------
# 1. Model selection — set once, globally. Every component picks these up.
# ---------------------------------------------------------------------------
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# ---------------------------------------------------------------------------
# 2. One knowledge base (VectorStoreIndex) per department
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

INDEXES = {
    dept: VectorStoreIndex.from_documents([Document(text=t) for t in texts])
    for dept, texts in DEPARTMENT_DOCS.items()
}


# ---------------------------------------------------------------------------
# 3. Custom events — these are the "wires" between steps
# ---------------------------------------------------------------------------
class RoutedEvent(Event):
    department: str


class ContextReadyEvent(Event):
    department: str
    context: str


# ---------------------------------------------------------------------------
# 4. The workflow — each @step declares what event it consumes and emits
# ---------------------------------------------------------------------------
class EmailRouterWorkflow(Workflow):
    @step
    async def classify(self, ctx: Context, ev: StartEvent) -> RoutedEvent:
        """Route the email to a department."""
        email: str = ev.email
        await ctx.store.set("email", email)  # stash for later steps

        prompt = (
            "Classify this customer email into exactly one department: "
            "billing, support, or sales.\n"
            "Reply with only the department name, lowercase.\n\n"
            f"EMAIL:\n{email}"
        )
        resp = await Settings.llm.acomplete(prompt)
        dept = resp.text.strip().lower()
        if dept not in INDEXES:  # fallback if the model gets creative
            dept = "support"
        return RoutedEvent(department=dept)

    @step
    async def retrieve(self, ctx: Context, ev: RoutedEvent) -> ContextReadyEvent:
        """RAG: pull the most relevant policy chunks from that department."""
        email: str = await ctx.store.get("email")
        retriever = INDEXES[ev.department].as_retriever(similarity_top_k=2)
        nodes = await retriever.aretrieve(email)
        context = "\n\n".join(n.get_content() for n in nodes)
        return ContextReadyEvent(department=ev.department, context=context)

    @step
    async def draft(self, ctx: Context, ev: ContextReadyEvent) -> StopEvent:
        """Draft the reply grounded in the retrieved context."""
        email: str = await ctx.store.get("email")
        prompt = (
            f"You are a {ev.department} representative. Draft a short, "
            "friendly reply to the customer email below. Base your answer "
            "ONLY on the company knowledge provided.\n\n"
            f"COMPANY KNOWLEDGE:\n{ev.context}\n\n"
            f"CUSTOMER EMAIL:\n{email}\n\n"
            "REPLY:"
        )
        resp = await Settings.llm.acomplete(prompt)
        return StopEvent(
            result={"department": ev.department, "draft": resp.text.strip()}
        )


# ---------------------------------------------------------------------------
# 5. Run it
# ---------------------------------------------------------------------------
async def main() -> None:
    wf = EmailRouterWorkflow(timeout=60, verbose=False)

    emails = [
        "Hi, I was charged twice this month and I'd like a refund for the "
        "duplicate charge. Order #4482.",
        "The app keeps showing error E-4012 whenever I open a project. Help!",
        "We're a 40-person non-profit evaluating your Team plan — do you "
        "offer any discounts?",
    ]

    for email in emails:
        result = await wf.run(email=email)
        print(f"\n=== routed to: {result['department'].upper()} ===")
        print(result["draft"])


if __name__ == "__main__":
    asyncio.run(main())
