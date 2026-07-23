"""
Email router + RAG drafter with a 3-reply cap — LlamaIndex Workflows version.

Business rule: the agent may reply to a customer thread at most 3 times.
On the 4th inbound email, it stops drafting and forwards the whole thread
to the department inbox for human review.

Event flow:

    StartEvent(email)
        -> intake  --(reply_count >= 3)--> EscalateEvent -> escalate -> StopEvent
                   --(else)-------------> RoutedEvent  -> retrieve
                                                          -> ContextReadyEvent
                                                          -> draft -> StopEvent

Per-customer threads persist by reusing the same Context object across
wf.run() calls (thread history, department, reply_count all live in
ctx.store; the Context is serializable for real deployments).

Run:
    pip install llama-index-core llama-index-llms-openai llama-index-embeddings-openai
    export OPENAI_API_KEY=...
    python email_router_llamaindex_v2.py
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

MAX_AGENT_REPLIES = 3

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

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

INDEXES = {
    dept: VectorStoreIndex.from_documents([Document(text=t) for t in texts])
    for dept, texts in DEPARTMENT_DOCS.items()
}


# ---------------------------------------------------------------------------
# Events — the wires between steps
# ---------------------------------------------------------------------------
class RoutedEvent(Event):
    department: str


class ContextReadyEvent(Event):
    department: str
    context: str


class EscalateEvent(Event):
    department: str


def _format_transcript(history: list[dict]) -> str:
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------
class EmailThreadWorkflow(Workflow):
    @step
    async def intake(
        self, ctx: Context, ev: StartEvent
    ) -> RoutedEvent | EscalateEvent:
        """Record the inbound email, enforce the reply cap, route."""
        email: str = ev.email

        history: list = await ctx.store.get("history", default=[])
        history.append({"role": "customer", "content": email})
        await ctx.store.set("history", history)

        reply_count: int = await ctx.store.get("reply_count", default=0)
        department: str | None = await ctx.store.get("department", default=None)

        # Reply cap reached -> hand the thread to humans
        if reply_count >= MAX_AGENT_REPLIES:
            return EscalateEvent(department=department or "support")

        # Classify once per thread; the department then sticks
        if department is None:
            resp = await Settings.llm.acomplete(
                "Classify this customer email into exactly one department: "
                "billing, support, or sales. Reply with only the department "
                f"name, lowercase.\n\nEMAIL:\n{email}"
            )
            department = resp.text.strip().lower()
            if department not in INDEXES:
                department = "support"
            await ctx.store.set("department", department)

        return RoutedEvent(department=department)

    @step
    async def retrieve(self, ctx: Context, ev: RoutedEvent) -> ContextReadyEvent:
        history: list = await ctx.store.get("history")
        latest = history[-1]["content"]
        retriever = INDEXES[ev.department].as_retriever(similarity_top_k=2)
        nodes = await retriever.aretrieve(latest)
        context = "\n\n".join(n.get_content() for n in nodes)
        return ContextReadyEvent(department=ev.department, context=context)

    @step
    async def draft(self, ctx: Context, ev: ContextReadyEvent) -> StopEvent:
        history: list = await ctx.store.get("history")
        resp = await Settings.llm.acomplete(
            f"You are a {ev.department} representative. Continue this email "
            "thread with a short, friendly reply to the customer's latest "
            "message. Base your answer ONLY on the company knowledge "
            "provided.\n\n"
            f"COMPANY KNOWLEDGE:\n{ev.context}\n\n"
            f"THREAD SO FAR:\n{_format_transcript(history)}\n\n"
            "REPLY:"
        )
        reply = resp.text.strip()

        history.append({"role": "agent", "content": reply})
        await ctx.store.set("history", history)
        count = await ctx.store.get("reply_count", default=0)
        await ctx.store.set("reply_count", count + 1)

        return StopEvent(
            result={"type": "reply", "department": ev.department, "text": reply}
        )

    @step
    async def escalate(self, ctx: Context, ev: EscalateEvent) -> StopEvent:
        history: list = await ctx.store.get("history")
        courtesy = (
            "Thanks for your patience — I've forwarded your thread to our "
            f"{ev.department} team, and a colleague will follow up with you "
            "directly."
        )
        history.append({"role": "agent", "content": courtesy})
        await ctx.store.set("history", history)

        forward = (
            f"To: {ev.department}@company.com\n"
            f"Subject: [ESCALATION] Thread exceeded {MAX_AGENT_REPLIES} "
            "automated replies — human review required\n\n"
            "The automated agent has reached its reply limit on this thread. "
            "Full transcript below.\n\n"
            f"{_format_transcript(history)}"
        )
        return StopEvent(
            result={
                "type": "escalated",
                "department": ev.department,
                "text": courtesy,
                "forward": forward,
            }
        )


# ---------------------------------------------------------------------------
# Simulate one customer thread: 4 inbound emails -> 3 replies + 1 escalation
# ---------------------------------------------------------------------------
async def main() -> None:
    wf = EmailThreadWorkflow(timeout=120, verbose=False)
    ctx = Context(wf)  # ONE context = ONE customer thread; reuse across runs

    inbound = [
        "Hi, I was charged twice this month and I'd like a refund for the "
        "duplicate charge. Order #4482.",
        "Thanks — how long will the refund take to show up on my card?",
        "It's been 2 days and I don't see it yet. Can you check?",
        "Still nothing. This is really frustrating, I need this resolved now.",
    ]

    for i, email in enumerate(inbound, 1):
        result = await wf.run(email=email, ctx=ctx)
        print(f"\n===== inbound #{i} =====")
        print(f"CUSTOMER: {email}")
        if result["type"] == "escalated":
            print(f"AGENT (auto-reply): {result['text']}")
            print("\n--- INTERNAL FORWARD ---")
            print(result["forward"])
        else:
            print(f"AGENT ({result['department']}): {result['text']}")


if __name__ == "__main__":
    asyncio.run(main())
