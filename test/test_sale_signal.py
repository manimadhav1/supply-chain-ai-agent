from agents.sales_signal_agent import (
    sales_signal_agent
)

state = {

    "sales_note":
    """
    Retailers reporting
    unusually high demand
    because of IPL Finals.
    """

}

result = sales_signal_agent(
    state
)

print(result)