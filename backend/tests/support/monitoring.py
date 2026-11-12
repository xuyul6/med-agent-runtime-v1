"""Helpers that bind runtimes to monitors and inject failing dependencies."""
from medagent.monitoring import Monitor, MonitorPolicy


class Clock:
    def __init__(self): self.now = 0.0
    def __call__(self): return self.now
    def advance(self, seconds=10): self.now += seconds


def monitor(policy=None):
    clock = Clock()
    return Monitor(policy or MonitorPolicy(), clock, lambda:1700000000+clock()), clock


def poison(monitor, name):
    for _ in range(monitor.policy.min_samples): monitor.record(name,False,100)
    monitor.sample(); monitor.sample()
    monitor.record(name,False,100)
    monitor.sample()
    assert monitor.agent_series[name].penalty >= monitor.policy.fallback_penalty


def bind(runtime, policy=None):
    monitored, clock = monitor(policy)
    for role,agents in runtime.agents.items():
        for agent in agents: monitored.register_agent(agent.instance, role)
    monitored.register_agent(runtime.risk_agent.instance,"risk_handoff",deterministic=True)
    for name,spec in runtime.tools.registry.specs.items(): monitored.register_tool(name,spec.dependency)
    runtime.monitor = runtime.tools.monitor = monitored
    monitored.circuits = runtime.tools.circuit_snapshot
    return monitored, clock
