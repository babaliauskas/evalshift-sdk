"""Self-contained EvalShift SDK demo agent.

A tiny *deterministic* support agent (no real LLM). Its only job is to emit a schema-valid
capture that drives the full lifecycle: capture -> `evalshift capture sync` -> golden suite
-> `evalshift run` -> `evalshift push`. See README.md for the full walkthrough.

Two things make a capture promotable:
  * ``record_model_call(input={"query": ...})`` -- promotion recovers the golden case ``inputs``
    from the first model call's input.
  * ``@capture.tool`` calls -- promotion turns the recorded tool calls into ``expected_tools``.

Run with capture on (writes ``.evalshift/captures/support_demo/cap_*.json``):

    EVALSHIFT_CAPTURE=1 python agent.py
"""

from __future__ import annotations

from evalshift import capture, record_model_call


@capture.tool(name="search_orders")
def search_orders(customer_id: str) -> dict:
    """Look up a customer's orders. Recorded as an expected tool on promote."""
    return {"orders": [{"id": "12345", "status": "delivered"}]}


@capture.tool(name="issue_refund")
def issue_refund(order_id: str) -> dict:
    """Issue a refund on an order."""
    return {"status": "refunded", "order_id": order_id}


# The tools the router model was actually offered on this call -- recorded alongside the model
# call itself (schema 2.0.0) so the CLI can tell "no tools offered" apart from "we don't know".
ROUTER_TOOLS = [
    {
        "name": "search_orders",
        "description": "Look up a customer's orders.",
        "input_schema": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Issue a refund on an order.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
]


@capture.agent(suite="support_demo", redact=True, tools=[])  # this agent never switches toolsets
def handle_ticket(query: str) -> str:
    """Resolve a support ticket. The model 'decides', then the agent calls tools."""
    # input carries the raw query so `capture promote` can recover the case inputs.
    # tools= is required here too -- it has no default at any entry point. The decorator's []
    # above only becomes this session's *inherited* toolset for a call that explicitly passes
    # tools=None; this call asserts its own list instead, which always wins.
    record_model_call(
        model_id="demo/router", tools=ROUTER_TOOLS, input={"query": query}, output="On it."
    )
    search_orders(customer_id="customer_42")
    if "refund" in query.lower():
        issue_refund(order_id="12345")
        return "Refund issued for order #12345."
    return "Your order is on its way."


if __name__ == "__main__":
    # Two tickets -> two captures -> a two-case golden suite.
    handle_ticket("I need a refund for order #12345 - customer_42, the item arrived damaged.")
    handle_ticket("Where is my order #67890? I'm customer_42.")
    print(
        "done -- set EVALSHIFT_CAPTURE=1 to write captures under .evalshift/captures/support_demo/"
    )
