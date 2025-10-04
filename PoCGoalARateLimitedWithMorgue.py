import os
import time
import random
import logging
import json
from datetime import datetime
import re  # Added for better JSON extraction
import numpy as np
import pandas as pd
from typing import List, Dict
import google.api_core.exceptions as google_exceptions # type: ignore
from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent # type: ignore

# Part 1: Core Components 

class Patient:
    """Represents a patient with attributes."""
    def __init__(self, id: int, severity: int, arrival_time: int):
        self.id = id
        self.severity = severity  # 1-10, lower better
        self.treatment_history = []  # List of treatments applied
        self.arrival_time = arrival_time
        self.last_treated = arrival_time
        self.deceased = False
        self.time_of_death = None

    def update_condition(self, treatment_effect: int = 0, untreated_penalty: int = 1):
        """Update severity based on treatment or time passage."""
        if self.deceased:
            return self.severity
        if treatment_effect:
            self.severity = max(1, self.severity + treatment_effect)  # Negative effect improves
        else:
            self.severity = min(10, self.severity + untreated_penalty)  # Worsens if untreated
        return self.severity

    def mark_deceased(self, time_of_death: int):
        self.deceased = True
        self.time_of_death = time_of_death

    def __str__(self):
        if self.deceased:
            return f"P{self.id}:Deceased"
        return f"P{self.id}:S{self.severity}"

class Scenario:
    """Manages the simulation environment."""
    def __init__(self, timeline_duration: int = 60, initial_patients: int = 5, supply_mode: str = "normal", logger: logging.Logger | None = None):
        self.timeline = 0
        self.max_timeline = timeline_duration  # In simulated minutes
        self.patients = self._generate_initial_patients(initial_patients)
        self.supplies = {"meds": 100, "beds": 10} if supply_mode == "low" else {"meds": float('inf'), "beds": float('inf')}
        self.last_events: List[Dict] = []  # List of events from last tick
        self.patient_queue = list(self.patients.keys())  # IDs in queue
        self.morgue: Dict[int, Patient] = {}
        self.logger = logger or logging.getLogger(__name__)
        self.death_probability = 0.1

    def _generate_initial_patients(self, num: int) -> Dict[int, Patient]:
        patients = {}
        for i in range(1, num + 1):
            severity = random.randint(4, 8)  # Moderate to severe
            patients[i] = Patient(i, severity, 0)
        return patients

    def tick(self):
        """Advance timeline by 1 minute, apply untreated penalties, generate events."""
        self.timeline += 1
        self.last_events = []
        for pid, patient in list(self.patients.items()):
            if patient.deceased:
                continue
            if self.timeline - patient.last_treated > 5:  # Worsen every 5 min untreated
                initial_sev = patient.severity
                patient.update_condition()
                post_sev = patient.severity
                if post_sev > initial_sev:
                    self.last_events.append({
                        "outcome": "worsening",
                        "patient_id": pid,
                        "initial_sev": initial_sev,
                        "post_sev": post_sev
                    })
            self._maybe_decease(pid, patient)
        self._generate_events()

    def _generate_events(self):
        """Generate random events: new patient (30% chance) or emergency (20% chance)."""
        if random.random() < 0.3:  # New patient
            new_id = max(self.patients.keys()) + 1 if self.patients else 1
            new_severity = random.randint(1, 10)
            self.patients[new_id] = Patient(new_id, new_severity, self.timeline)
            self.patient_queue.append(new_id)
            self.logger.info(f"Event: New patient {new_id} arrived with severity {new_severity}")
        if random.random() < 0.2:  # Emergency
            if self.patients:
                pid = random.choice(list(self.patients.keys()))
                patient = self.patients[pid]
                if not patient.deceased:
                    initial_sev = patient.severity
                    patient.update_condition(untreated_penalty=2)  # Worsen faster
                    post_sev = patient.severity
                    if post_sev > initial_sev:
                        self.last_events.append({
                            "outcome": "worsening",
                            "patient_id": pid,
                            "initial_sev": initial_sev,
                            "post_sev": post_sev
                        })
                    self._maybe_decease(pid, patient)
                    self.logger.info(f"Event: Emergency for patient {pid}, severity now {patient.severity}")

    def get_state(self) -> str:
        """Concise string representation of current state for agents (reduced tokens)."""
        patient_str = ",".join([str(p) for p in self.patients.values()])
        supply_str = f"Meds:{self.supplies['meds']},Beds:{self.supplies['beds']}"
        queue_str = ",".join(map(str, self.patient_queue))
        morgue_str = ",".join([str(p) for p in self.morgue.values()])
        base = f"T:{self.timeline}/{self.max_timeline}|Pts:{patient_str}|Sup:{supply_str}|Q:{queue_str}"
        if morgue_str:
            base = f"{base}|Morgue:{morgue_str}"
        return base

    def apply_action(self, action: Dict) -> Dict | None:
        """Apply agent's action: e.g., {'treat': pid, 'with': 'meds', 'effect': -2}"""
        if 'treat' in action:
            pid = self._normalize_patient_id(action['treat'])
            if pid in self.patients:
                patient = self.patients[pid]
                if action.get('with') == 'meds' and self.supplies['meds'] > 0:
                    self.supplies['meds'] -= 1
                    effect = action.get('effect', -2)
                    initial_sev = patient.severity
                    new_sev = patient.update_condition(effect)
                    patient.last_treated = self.timeline
                    patient.treatment_history.append(action)
                    if new_sev < 3:  # Discharge threshold
                        self._discharge_patient(pid)
                        return {"outcome": "discharge", "patient_id": pid, "last_sev": new_sev}
                    if self._maybe_decease(pid, patient):
                        return {"outcome": "death", "patient_id": pid}
                    outcome = "improvement" if effect < 0 else "worsening"
                    return {
                        "outcome": outcome,
                        "patient_id": pid,
                        "initial_sev": initial_sev,
                        "post_sev": new_sev
                    }
        return None

    def get_last_events(self) -> List[Dict]:
        events = self.last_events
        self.last_events = []
        return events

    def is_ended(self) -> bool:
        return self.timeline >= self.max_timeline or not self.patients

    def _normalize_patient_id(self, raw_pid) -> int | None:
        if isinstance(raw_pid, int):
            return raw_pid
        if isinstance(raw_pid, str):
            raw_pid = raw_pid.strip().upper()
            if raw_pid.startswith("P"):
                raw_pid = raw_pid[1:]
            if raw_pid.isdigit():
                return int(raw_pid)
        return None

    def _discharge_patient(self, pid: int):
        if pid in self.patients:
            del self.patients[pid]
        self.patient_queue = [p for p in self.patient_queue if p != pid]

    def _move_to_morgue(self, pid: int, patient: Patient):
        patient.mark_deceased(self.timeline)
        self.morgue[pid] = patient
        if pid in self.patients:
            del self.patients[pid]
        self.patient_queue = [p for p in self.patient_queue if p != pid]
        self.logger.warning(f"Patient {pid} has died and moved to the morgue.")
        self.last_events.append({"outcome": "death", "patient_id": pid})

    def _maybe_decease(self, pid: int, patient: Patient) -> bool:
        if patient.deceased:
            return True
        if patient.severity >= 8 and random.random() < self.death_probability:
            self._move_to_morgue(pid, patient)
            return True
        return False

class Scorer:
    """Tracks grading points."""
    def __init__(self):
        self.points = 0
        self.history = []

    def update(self, event_dict: Dict, tick: int, nurse: str | None = None):
        outcome = event_dict["outcome"]
        pid = event_dict["patient_id"]
        init_sev = event_dict.get("initial_sev")
        post_sev = event_dict.get("post_sev")
        last_sev = event_dict.get("last_sev")

        if outcome == "improvement":
            log = f"Treated [{tick}, {nurse}, {pid}, {init_sev}, {post_sev}]"
            self.points += 1
        elif outcome == "worsening":
            log = f"Worsened - [{tick}, {pid}, {init_sev}, {post_sev}]"
            self.points -= 1 if nurse else 0  # Deduct only for action-based worsening
        elif outcome == "discharge":
            log = f"Discharged - [{tick}, {pid}, {last_sev if last_sev else init_sev}]"
            self.points += 2
        elif outcome == "death":
            log = f"Deceased - [{tick}, {pid}]"
            self.points -= 3
        else:
            return  # Invalid outcome

        self.history.append(log)

    def get_score(self) -> int:
        return self.points

    def __str__(self):
        return f"Score: {self.points} | History: {self.history}"

# Part 2: Agent Setup with AutoGen

def setup_agents(num_nurses: int = 3) -> List[AssistantAgent]:
    """Create nurse agents with Gemini model."""
    # Load config for Gemini with caching
    config_list = [
        {
            "model": "gemini-1.5-pro",
            "api_key": os.getenv("GOOGLE_API_KEY", "AIzaSyCv7CHW4E7atp6QW6WlySwLpV0l1HU1Ebc"),
            "api_type": "google"
        }
    ]
    llm_config = {"config_list": config_list, "cache_seed": 42}  # Enable caching

    system_message = """You are a nurse. Assess pts, treat, coord no overlaps. Prior high sev.
Action: JSON {"treat": id, "with": "meds", "effect": -2} or "Pass".
One action/turn. Obs state & msgs."""  # Shortened for token savings

    agents = []
    for i in range(1, num_nurses + 1):
        agent = AssistantAgent(
            name=f"Nurse_{i}",
            system_message=system_message,
            llm_config=llm_config
        )
        agents.append(agent)

    return agents

# Part 3: Simulation Loop with Optimizations

def setup_logging() -> tuple[logging.Logger, str]:
    logger = logging.getLogger("triage_sim")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()

    log_filename = datetime.now().strftime("%H%M%S.%Y-%m-%d_log.txt")
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_filename)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    logger.info(f"Logging initialized. Output file: {log_path}")
    return logger, log_path

def extract_action(message: str) -> Dict | None:
    if not message:
        return None
    # Improved extraction: Use regex to find the JSON-like string in the message
    json_match = re.search(r'\{.*?\}', message, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    return None

def run_simulation(num_nurses: int = 3, timeline_duration: int = 60, supply_mode: str = "normal"):
    """Main simulation for Goal A with rate/token optimizations."""
    logger, log_path = setup_logging()
    scenario = Scenario(timeline_duration=timeline_duration, supply_mode=supply_mode, logger=logger)
    scorer = Scorer()
    agents = setup_agents(num_nurses)

    user_proxy = UserProxyAgent(
        name="Admin",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    group_chat = GroupChat(agents=agents, messages=[], max_round=num_nurses, speaker_selection_method="round_robin")
    manager = GroupChatManager(groupchat=group_chat, llm_config={"config_list": [{"model": "gemini-1.5-pro", "api_key": os.getenv("GOOGLE_API_KEY", "AIzaSyCv7CHW4E7atp6QW6WlySwLpV0l1HU1Ebc"), "api_type": "google"}], "cache_seed": 42})

    last_request_time: float | None = None

    while not scenario.is_ended():
        if last_request_time is not None:
            elapsed = time.time() - last_request_time
            if elapsed < 30:
                time.sleep(30 - elapsed)
        state_msg = f"State:{scenario.get_state()}\nDiscuss actions."  # Concise
        try:
            user_proxy.initiate_chat(manager, message=state_msg)
            last_request_time = time.time()
        except google_exceptions.ResourceExhausted as e:
            logger.warning(f"Rate limit hit: {e}")
            retry_delay = 60  # Safe default for free tier
            time.sleep(retry_delay)
            continue
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            time.sleep(30)  # Backoff
            continue

        # Process actions from chat history
        for msg in group_chat.messages[-num_nurses:]:
            if "content" in msg and isinstance(msg["content"], str):
                action = extract_action(msg["content"])
                if action:
                    outcome_dict = scenario.apply_action(action)
                    if outcome_dict:
                        nurse = msg.get("name", "Unknown")
                        scorer.update(outcome_dict, scenario.timeline, nurse)
                        logger.info(f"Outcome for {action}: {outcome_dict['outcome']}")

        group_chat.messages = []  # Clear history to reduce tokens in next chat

        scenario.tick()
        for event_dict in scenario.get_last_events():
            scorer.update(event_dict, scenario.timeline)
        time.sleep(30)  # Increased throttle to 30s to avoid 429 errors (was 15s)

    logger.info(f"Simulation ended. Final Score: {scorer}")
    logger.info(f"Detailed log saved to {log_path}")

if __name__ == "__main__":
    run_simulation(num_nurses=3, timeline_duration=30, supply_mode="normal")