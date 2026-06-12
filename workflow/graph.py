from langgraph.graph import StateGraph

from agents.sales_signal_agent import (
    sales_signal_agent
)
from agents.summary_agent import (
    summary_agent   
)
from workflow.state import (
    SupplyChainState
)

from agents.demand_agent import (
    demand_agent
)

from agents.inventory_agent import (
    inventory_agent
)

from agents.risk_agent import (
    risk_agent
)

from agents.procurement_agent import (
    procurement_agent
)
#added by {neeraj_k} for rag:
#deleted^

workflow = StateGraph(
    SupplyChainState
)

workflow.add_node(
    "demand",
    demand_agent
)

workflow.add_node(
    "inventory",
    inventory_agent
)

workflow.add_node(
    "risk",
    risk_agent
)

workflow.add_node(
    "procurement",
    procurement_agent
)

workflow.add_node(
    "sales_signal",
    sales_signal_agent
)

workflow.add_node(
    "summary",
    summary_agent
)

workflow.add_edge(
    "procurement",
    "summary"
)
#sales signal changed by neeraj k
workflow.add_edge(
    "sales_signal",
    "demand"
)
workflow.add_edge(
    "demand",
    "inventory"
)

workflow.add_edge(
    "inventory",
    "risk"
)

workflow.add_edge(
    "risk",
    "procurement"
)

workflow.set_entry_point(
    "sales_signal"
)

workflow.set_finish_point(
    "summary"
)

graph = workflow.compile()