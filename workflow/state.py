from typing_extensions import TypedDict

class AgentState(TypedDict):
    sku: str
    region: str
    temp: int
    promo: int
    event: int
    sales_note: str
    forecast: float  # Adjusted to float to match SupplyChainState if needed
    inventory: int
    shortage: int
    days_cover: float
    risk: str
    reorder_qty: int
    demand_boost: float
    summary: str
    kb_context: str  # ◄── New RAG Field added here

class SupplyChainState(TypedDict):
    sku: str
    region: str
    temp: int
    promo: int
    event: int
    month: int       # Unique to SupplyChainState
    dayofweek: int   # Unique to SupplyChainState
    sales_note: str
    forecast: float
    inventory: int
    shortage: int
    days_cover: float
    risk: str
    reorder_qty: int
    demand_boost: float
    summary: str
    kb_context: str  # ◄── Added here as well to keep them in sync