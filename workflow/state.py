from typing import TypedDict


class SupplyChainState(TypedDict):

    sku: str
    region: str

    temp: int
    promo: int
    event: int

    month: int
    dayofweek: int

    sales_note: str

    forecast: float

    inventory: int

    shortage: int

    days_cover: float

    risk: str

    reorder_qty: int

    demand_boost: float

    summary: str

