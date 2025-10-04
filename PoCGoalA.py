# poc_mas_goal_a.py
# Proof-of-Concept for Multi-Agent System (Goal A) using AutoGen with Gemini-1.5-pro
# Note: User specified 'gemini-2.5-pro', but as of current knowledge, Gemini models are like 'gemini-1.5-pro'.
#       Assuming it's a future/typo for 'gemini-1.5-pro'. Adjust model name if needed.
#       Requires Google API key for Gemini. Set environment variable: export GOOGLE_API_KEY="your_key"

import os
import time
import random
import numpy as np
import pandas as pd
from typing import List, Dict
from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent, config_list_from_json # type: ignore

# Part 1: Define Core Components

class Patient:
    """Represents a patient with attributes."""
    def __init__(self, id: int, severity: int, arrival_time: int):
        self.id = id
        self.severity = severity  # 1-10, lower better
        self.treatment_history = []  # List of treatments applied
        self.arrival_time = arrival_time
        self.last_treated = arrival_time

    def update_condition(self, treatment_effect: int = 0, untreated_penalty: int = 1):
        """Update severity based on treatment or time passage."""
        if treatment_effect:
            self.severity = max(1, self.severity + treatment_effect)  # Negative effect improves
        else:
            self.severity = min(10, self.severity + untreated_penalty)  # Worsens if untreated
        return self.severity

    def __str__(self):
        return f"Patient {self.id}: Severity {self.severity}"

class Scenario:
    """Manages the simulation environment."""
    def __init__(self, timeline_duration: int = 60, initial_patients: int = 5, supply_mode: str = "normal"):
        self.timeline = 0
        self.max_timeline = timeline_duration  # In simulated minutes
        self.patients = self._generate_initial_patients(initial_patients)
        self.supplies = {"meds": 100, "beds": 10} if supply_mode == "low" else {"meds": float('inf'), "beds": float('inf')}
        self.events = []  # List of pending events
        self.patient_queue = list(self.patients.keys())  # IDs in queue

    def _generate_initial_patients(self, num: int) -> Dict[int, Patient]:
        patients = {}
        for i in range(1, num + 1):
            severity = random.randint(4, 8)  # Moderate to severe
            patients[i] = Patient(i, severity, 0)
        return patients

    def tick(self):
        """Advance timeline by 1 minute, apply untreated penalties, generate events."""
        self.timeline += 1
        for patient in self.patients.values():
            if self.timeline - patient.last_treated > 5:  # Worsen every 5 min untreated
                patient.update_condition()
        self._generate_events()

    def _generate_events(self):
        """Generate random events: new patient (30% chance) or emergency (20% chance)."""
        if random.random() < 0.3:  # New patient
            new_id = max(self.patients.keys()) + 1 if self.patients else 1
            new_severity = random.randint(1, 10)
            self.patients[new_id] = Patient(new_id, new_severity, self.timeline)
            self.patient_queue.append(new_id)
            print(f"Event: New patient {new_id} arrived with severity {new_severity}")
        if random.random() < 0.2:  # Emergency
            if self.patients:
                pid = random.choice(list(self.patients.keys()))
                self.patients[pid].update_condition(untreated_penalty=2)  # Worsen faster
                print(f"Event: Emergency for patient {pid}, severity now {self.patients[pid].severity}")

    def get_state(self) -> str:
        """Return string representation of current state for agents."""
        patient_str = "\n".join([str(p) for p in self.patients.values()])
        supply_str = f"Supplies: {self.supplies}"
        return f"Timeline: {self.timeline}/{self.max_timeline}\nPatients:\n{patient_str}\n{supply_str}\nQueue: {self.patient_queue}"

    def apply_action(self, action: Dict):
        """Apply agent's action: e.g., {'treat': pid, 'with': 'meds', 'effect': -2}"""
        if 'treat' in action:
            pid = action['treat']
            if pid in self.patients:
                patient = self.patients[pid]
                if action.get('with') == 'meds' and self.supplies['meds'] > 0:
                    self.supplies['meds'] -= 1
                    effect = action.get('effect', -2)  # Improve by 2
                    new_sev = patient.update_condition(effect)
                    patient.last_treated = self.timeline
                    patient.treatment_history.append(action)
                    if new_sev < 3:  # Discharge threshold
                        del self.patients[pid]
                        self.patient_queue = [p for p in self.patient_queue if p != pid]
                        return "discharge"
                    elif new_sev > 8:  # Risk of death
                        if random.random() < 0.1:  # 10% chance to die if >8
                            del self.patients[pid]
                            self.patient_queue = [p for p in self.patient_queue if p != pid]
                            return "death"
                    return "improvement" if effect < 0 else "worsening"
        return None

    def is_ended(self) -> bool:
        return self.timeline >= self.max_timeline or not self.patients

class Scorer:
    """Tracks grading points."""
    def __init__(self):
        self.points = 0
        self.history = []

    def update(self, outcome: str):
        if outcome == "improvement":
            self.points += 1
        elif outcome == "discharge":
            self.points += 2
        elif outcome == "worsening":
            self.points -= 1
        elif outcome == "death":
            self.points -= 3
        self.history.append(outcome)

    def get_score(self) -> int:
        return self.points

    def __str__(self):
        return f"Score: {self.points} | History: {self.history}"

# Part 2: Agent Setup with AutoGen

def setup_agents(num_nurses: int = 3) -> List[AssistantAgent]:
    """Create nurse agents with Gemini model."""
    # Load config for Gemini
    config_list = [
        {
            "model": "gemini-2.5-pro",
            "api_key": "AIzaSyCv7CHW4E7atp6QW6WlySwLpV0l1HU1Ebc",
            "api_type": "google"
        }
    ]

    system_message = """
You are a nurse in a hospital simulation. Your role: Assess patients, decide treatments, coordinate with other nurses to avoid overlaps. Prioritize emergencies (high severity).
Actions: Propose to treat a patient, e.g., 'Treat patient X with meds for -2 effect'.
State your action clearly in JSON format: {"treat": patient_id, "with": "meds", "effect": -2}
Only one action per turn. If no action needed, say "Pass".
Observe the current state and previous messages.
"""

    agents = []
    for i in range(1, num_nurses + 1):
        agent = AssistantAgent(
            name=f"Nurse_{i}",
            system_message=system_message,
            llm_config={"config_list": config_list}
        )
        agents.append(agent)

    return agents

# Part 3: Simulation Loop

def run_simulation(num_nurses: int = 3, timeline_duration: int = 60, supply_mode: str = "normal"):
    """Main simulation for Goal A."""
    scenario = Scenario(timeline_duration=timeline_duration, supply_mode=supply_mode)
    scorer = Scorer()
    agents = setup_agents(num_nurses)

    # Add a user proxy to initiate (simulates admin)
    user_proxy = UserProxyAgent(
        name="Admin",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    group_chat = GroupChat(agents=agents, messages=[], max_round=10)  # Limit rounds per tick
    manager = GroupChatManager(groupchat=group_chat, llm_config={"config_list": [{"model": "gemini-2.5-pro", "api_key": "AIzaSyCv7CHW4E7atp6QW6WlySwLpV0l1HU1Ebc", "api_type": "google"}]})

    while not scenario.is_ended():
        state_msg = f"Current State:\n{scenario.get_state()}\nDiscuss and decide actions."
        user_proxy.initiate_chat(manager, message=state_msg)

        # Process actions from chat history (parse last messages)
        for msg in group_chat.messages[-num_nurses:]:  # Assume last N messages are actions
            if "content" in msg and isinstance(msg["content"], str):
                try:
                    action = eval(msg["content"])  # Assuming JSON-like str
                    outcome = scenario.apply_action(action)
                    if outcome:
                        scorer.update(outcome)
                        print(f"Outcome for {action}: {outcome}")
                except:
                    pass  # Invalid action

        scenario.tick()
        time.sleep(1)  # Simulate real-time delay

    print(f"Simulation ended. Final Score: {scorer}")

if __name__ == "__main__":
    # Run PoC
    run_simulation(num_nurses=3, timeline_duration=30, supply_mode="normal")