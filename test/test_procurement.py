from agents.procurement_agent import procurement_agent

state = {
    "forecast":8000,
    "inventory":4000
}

print(
    procurement_agent(state)
)