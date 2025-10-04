# Multi-Agent System for Simulating Nursing Operations Using AutoGen

## Introduction
As a solo AI researcher, I developed a multi-agent system (MAS) in Python using AutoGen to simulate nursing operations in dynamic medical scenarios. Nurse agents manage patients with evolving severities (1-10 scale), responding to events like new arrivals or emergencies under resource constraints. The simulation operates on a timeline (e.g., 15 ticks), with performance scored as: +1 for condition improvements, +2 for discharges, -1 for worsenings, and -3 for deaths. This proof-of-concept (PoC) focuses on Goal A (independent local simulations) to validate feasibility.

## System Design
- **Core Components** (from core.py): Patients are objects with attributes like severity, treatment history, and a 10% death probability at severity ≥8. The Scenario class manages the timeline, events (30% chance new patient, 20% chance emergency), supplies (normal/infinite mode), and state updates.
- **Agents** (from agents.py): Three Nurse AssistantAgents, powered by Gemini-1.5-Flash LLM, coordinate via GroupChat in a round-robin setup. They output JSON actions (e.g., {"treat": "P2", "with": "meds", "effect": -2}) or pass, prioritizing high-severity patients and avoiding treatment overlaps.
- **Simulation Loop** (from simulation.py): Advances the timeline, triggers agent interactions with concise state messages, applies actions, logs events, and updates scores. It includes rate limit handling and token optimization by periodically clearing chat history.

## Goals and Current Work
- **Goal A: Independent Simulations**: Each computer runs the full scenario locally, with nurse agents collaborating via in-memory GroupChat. Grades are computed independently. This PoC implements Goal A on a single machine, testing agent coordination and system mechanics. **Pros**: Simple to implement, low-latency, ideal for parallel experiments and strategy benchmarking. **Cons**: Lacks inter-machine collaboration, limiting realism for distributed team dynamics.
- **Goal B: Distributed Agents**: Each computer hosts one or more agents, communicating across a network (e.g., via sockets or REST APIs) to coordinate actions like patient assignments. Grades reflect shared outcomes. **Pros**: Models real-world team dynamics and network delays, enabling study of emergent behaviors. **Cons**: Complex networking, potential synchronization issues, and higher latency. Goal B is planned but not yet implemented.
- **Current Work**: The PoC tests Goal A with 3 nurses, a 15-tick timeline, and normal (infinite) supplies. The recent run (log 194030.2025-10-04_log.txt) started with 5 patients (severities 5-8), achieving a score of 23 (13 treatments, 8 discharges, 2 worsenings, no deaths). The simulation ended early at Tick 9 with all patients discharged. Agents coordinated effectively, avoiding overlaps, with one initial 503 error resolved via rate limit handling.

## Future Work
- Implement Goal B using sockets for cross-machine agent communication.
- Integrate a central database (e.g., Redis) for shared state in distributed setups.
- Enhance patient dynamics with Markov chains for realistic condition evolution.
- Test low-supply mode to simulate resource constraints.
- Add agent specialization (e.g., emergency vs. routine nurses) and an ethics agent for oversight.
- Develop a CLI or dashboard (e.g., Streamlit) for real-time monitoring.
- Explore bioethics studies by simulating decision biases in resource allocation.
- I’m eager to collaborate on integrating real-time data or extending to multi-hospital scenarios!