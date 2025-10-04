# Multi-Agent System for Simulating Nursing Operations Using AutoGen

## Introduction
Using the clues you have provided me with ChatDev and the discussion we had, I found a framework called ["AutoGen" by Microsoft](https://github.com/microsoft/autogen) which can support both Local LLM and Commercial AI via API Token. A bit of searching for an LLM with free API Token led me to Google AI Studio, which provide me with 2500 API calls a day (or a month? I have to check this again), **gemini-2.5-flash** from them should be more than sufficient for this simple program.

For the initial design of this, we have: 
- Nurse agents manage patients with varying and worsening severities (1-10 scale, with 10 being worst, and severity increases every few days untreated), responding to events.
- Randomised events like new arrivals or emergencies under resource constraints (resource is infinite for current test run). 
- The simulation operates on a timeline (e.g., 15 ticks), with performance scored at the end as: +1 for condition improvements, +2 for discharges, -1 for worsenings, and -3 for deaths. This proof-of-concept (PoC) focuses on Goal A (independent local simulations) to check feasibility.
- Automated logging for all events and timeline for assessment.

## System Design
- **Core Components** (from core.py): Patients are objects with attributes like severity, treatment history, and a 10% death probability at severity ≥8. The Scenario class manages the timeline, events (30% chance new patient, 20% chance emergency), supplies (normal/infinite mode), and state updates.
- **Agents** (from agents.py): 2 Nurse AssistantAgents, running Gemini-2.5-Flash LLM, coordinate via GroupChat in a round-robin setup. They output JSON actions (e.g., {"treat": "P2", "with": "meds", "effect": -2}) or pass, prioritizing high-severity patients and avoiding treatment overlaps, pass only happens if there is no unoccupied patient left to treat or the timeline ends.
- **Simulation Loop** (from simulation.py): Advances the timeline, triggers agent interactions with concise state messages, applies actions, logs events, and updates scores. It includes rate limit handling and token optimization by periodically clearing chat history every 2 ticks.

## Goals and Current Work
- **Goal A: Independent Simulations**: Each computer runs the full scenario locally, with nurse agents collaborating via in-memory GroupChat. Grades are computed independently. This PoC implements Goal A on a single machine, testing agent coordination and system mechanics. **Pros**: Simple to implement, low-latency, ideal for parallel experiments and strategy benchmarking. **Cons**: Lacks inter-machine collaboration, limiting realism for distributed team dynamics.
- **Goal B: Distributed Agents**: Each computer hosts one or more agents, communicating across a network (e.g., via sockets or REST APIs) to coordinate actions like patient assignments. Score are measured by personal and total outcomes. **Pros**: Models real-world team dynamics and network delays, enabling study of emergent behaviors. **Cons**: Complex networking, potential synchronization issues, and higher latency. Goal B is planned but not yet implemented.
- **Current Work**: The PoC tests Goal A with 3 nurses, a 15-tick timeline, and infinite supplies. The recent run (log 194030.2025-10-04_log.txt) started with 5 patients (severities 5-8), achieving a score of 23 (13 treatments, 8 discharges, 2 worsenings, no deaths). The simulation ended early at Tick 9 with all patients discharged. Agents coordinated effectively, avoiding overlaps, with one initial 503 error resolved via rate limit handling.

## Future Work
- Implement more sudden events (mass casualty emergency, infectious disease patient.)
- Implement Goal B using sockets for cross-machine agent communication.
- Enhance patient dynamics with Markov chains for realistic condition evolution.
- Test low-supply mode to simulate resource constraints.
- Add agent specialization (e.g., emergency vs. routine nurses.)
- Develop a better visualisation for real-time monitoring (rather than the console/textbased log.)
- Explore bioethics studies by simulating decision biases in resource allocation.
- Integrating real-time data or extending to multi-hospital scenarios (e.g., allow hospitals to send patient to another hospital when running out of bed.)