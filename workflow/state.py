from typing import TypedDict


class SupplyChainState(TypedDict):
    # Product & region
    sku: str          # e.g. "SKU-003"
    region: str

    # Date/time features
    month: int
    dayofweek: int
    is_weekend: int

    # Environmental
    temperature: int

    # Promotion
    promo_type: int          # 0=none 1=discount_low 2=discount_high 3=bogo 4=bundle
    promo_discount_pct: int  # 0-50

    # External events
    event_type: int              # 0=none 1=festival 2=sports 3=national_holiday 4=wedding
    is_competitor_stockout: int  # 0 or 1

    # Qualitative notes for RAG / sales signal agent
    sales_note: str

    # Agent outputs
    demand_boost: float  # % uplift from sales signal parsing
    forecast: float
    inventory: int
    shortage: int
    days_cover: float
    risk: str
    reorder_qty: int
    summary: str
