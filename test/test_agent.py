from agents.demand_agent import demand_agent
from agents.inventory_agent import inventory_agent

state = {
    "sku":"Coke500",
    "region":"Bangalore",

    "temp":38,
    "promo":1,
    "event":1,

    "month":6,
    "dayofweek":5
}

state = demand_agent(state)

state = inventory_agent(state)

print(state)