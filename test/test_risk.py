from agents.risk_agent import risk_agent

state = {
    "forecast":8000,
    "inventory":4000
}

print(
    risk_agent(state)
)