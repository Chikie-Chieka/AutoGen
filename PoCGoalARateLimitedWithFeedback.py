import os
import time
import random
import logging
import json
from datetime import datetime
import re  # Added for better JSON extraction
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
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
        self.severity = -1  # Indicate deceased
        self.time_of_death = time_of_death

    def __str__(self):
        if self.deceased:
            return f"P{self.id}:Deceased"
        return f"P{self.id}:S{self.severity}"

class Scenario:
    """Manages the simulation environment."""
    def __init__(self, timeline_duration: int = 60, initial_patients: int = 5, supply_mode: str = "normal", logger: Optional[logging.Logger] = None):
        self.timeline = 0
        self.max_timeline = timeline_duration  # In simulated minutes
        self.patients = self._generate_initial_patients(initial_patients)
        self.supplies = {"meds": 100, "beds": 10} if supply_mode == "low" else {"meds": float('inf'), "beds": float('inf')}
        self.events = []  # List of pending events
        self.patient_queue = list(self.patients.keys())  # IDs in queue
        self.morgue: Dict[int, Patient] = {}
        self.logger = logger or logging.getLogger(__name__)
        self.death_probability = 0.1
        self.last_events = []  # Track events for the current tick

    def _generate_initial_patients(self, num: int) -> Dict[int, Patient]:
        patients = {}
        for i in range(1, num + 1):
            severity = random.randint(1, 8)  # Moderate to severe
            patients[i] = Patient(i, severity, 0)
        return patients

    def tick(self):
        """Advance timeline by 1 minute, apply untreated penalties, generate events."""
        self.timeline += 1
        self.last_events = []  # Clear events for this tick
        
        for pid, patient in list(self.patients.items()):
            if patient.deceased:
                continue
            if self.timeline - patient.last_treated > 5:  # Worsen every 5 min untreated
                patient.update_condition()
            if self._maybe_decease(pid, patient):
                continue
        self._generate_events()

    def _generate_events(self):
        """Generate random events: new patient (30% chance) or emergency (20% chance)."""
        if random.random() < 0.2:  # New patient
            new_id = max(self.patients.keys()) + 1 if self.patients else 1
            new_severity = random.randint(1, 10)
            self.patients[new_id] = Patient(new_id, new_severity, self.timeline)
            self.patient_queue.append(new_id)
            event_msg = f"Event: New patient {new_id} arrived with severity {new_severity}"
            self.logger.info(event_msg)
            self.last_events.append({
                "type": "new_patient",
                "patient_id": new_id,
                "severity": new_severity,
                "message": event_msg
            })
            
        if random.random() < 0.1:  # Emergency
            if self.patients:
                pid = random.choice(list(self.patients.keys()))
                patient = self.patients[pid]
                if not patient.deceased:
                    patient.update_condition(untreated_penalty=2)  # Worsen faster
                    if self._maybe_decease(pid, patient):
                        return
                    event_msg = f"Event: Emergency for patient {pid}, severity now {patient.severity}"
                    self.logger.info(event_msg)
                    self.last_events.append({
                        "type": "emergency",
                        "patient_id": pid,
                        "severity": patient.severity,
                        "message": event_msg
                    })

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
        
    def get_last_events(self):
        """Return events that happened in the last tick."""
        return self.last_events

    def apply_action(self, action: Dict):
        """Apply agent's action: e.g., {'treat': pid, 'with': 'meds', 'effect': -2}"""
        if 'treat' in action:
            pid = self._normalize_patient_id(action['treat'])
            if pid is not None and pid in self.patients:
                patient = self.patients[pid]
                if action.get('with') == 'meds' and self.supplies['meds'] > 0:
                    self.supplies['meds'] -= 1
                    effect = action.get('effect', -2)  # Improve by 2
                    new_sev = patient.update_condition(effect)
                    patient.last_treated = self.timeline
                    patient.treatment_history.append(action)
                    if new_sev < 3:  # Discharge threshold
                        self._discharge_patient(pid)
                        return "discharge"
                    elif self._maybe_decease(pid, patient):
                        return "death"
                    return "improvement" if effect < 0 else "worsening"
        return None

    def is_ended(self) -> bool:
        return self.timeline >= self.max_timeline or not self.patients

    def _normalize_patient_id(self, raw_pid) -> Optional[int]:
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
    # Load config for Gemini with caching
    config_list = [
        {
            "model": "gemini-2.5-flash",
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

def setup_logging() -> Tuple[logging.Logger, str]:
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

def extract_action(message: str) -> Optional[Dict]:
    if not message:
        return None
    # Improved extraction: Use regex to find the JSON-like string in the message
    json_match = re.search(r'\{.*?\}', message, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            action = json.loads(json_str)
            # Validate action structure
            if "treat" in action and "with" in action and "effect" in action:
                return action
            elif "pass" in action and action["pass"] is True:
                return {"pass": True}
        except json.JSONDecodeError:
            pass
    return None

class NurseAdminAlternator:
    """Helper class to manage alternating between nurses and admin."""
    def __init__(self, nurses, admin):
        self.nurses = nurses
        self.admin = admin
        self.current_nurse_idx = 0
        self.is_admin_turn = False
        
    def get_next_speaker(self):
        if self.is_admin_turn:
            self.is_admin_turn = False
            self.current_nurse_idx = (self.current_nurse_idx + 1) % len(self.nurses)
            return self.admin
        else:
            self.is_admin_turn = True
            return self.nurses[self.current_nurse_idx]
            
    def reset(self):
        self.current_nurse_idx = 0
        self.is_admin_turn = False

def create_admin_response(scenario, action, outcome, nurse_name):
    """Create a response from the admin after a nurse action."""
    if action is None:
        return f"Invalid action format from {nurse_name}. Please provide a valid JSON action."
        
    admin_response = ""
    
    # For pass action
    if "pass" in action and action["pass"]:
        admin_response = f"Action acknowledged. {nurse_name} chose to pass this turn."
    elif outcome is not None:
        # For treatment actions
        patient_id = action.get("treat")
        if outcome == "improvement":
            admin_response = f"Action acknowledged. You've treated Patient {patient_id} and their condition has improved."
        elif outcome == "discharge":
            admin_response = f"Action acknowledged. Excellent work! Patient {patient_id} has improved enough to be discharged."
        elif outcome == "worsening":
            admin_response = f"Action acknowledged. Despite treatment, Patient {patient_id}'s condition has worsened."
        elif outcome == "death":
            admin_response = f"Action acknowledged. Unfortunately, Patient {patient_id} did not survive treatment and has been moved to the morgue."
    else:
        admin_response = f"Action acknowledged, but no change in patient status."
    
    # Append current state
    admin_response += f"\n\nState:{scenario.get_state()}\nDiscuss actions."
    
    return admin_response

def run_simulation(num_nurses: int = 3, timeline_duration: int = 60, supply_mode: str = "normal"):
    """Main simulation for Goal A with rate/token optimizations."""
    logger, log_path = setup_logging()
    scenario = Scenario(timeline_duration=timeline_duration, supply_mode=supply_mode, logger=logger)
    scorer = Scorer()
    
    # Set up nurses
    nurse_agents = setup_agents(num_nurses)
    
    # Set up admin (no LLM needed - we'll provide responses programmatically)
    admin_agent = AssistantAgent(
        name="Admin",
        system_message="You are the hospital administrator providing feedback on nurse actions.",
        llm_config=None
    )
    
    # Set up alternator to manage turns
    alternator = NurseAdminAlternator(nurse_agents, admin_agent)
    
    # All agents (nurses + admin) for the group chat
    all_agents = nurse_agents + [admin_agent]
    
    # Set up group chat with alternating pattern
    group_chat = GroupChat(
        agents=all_agents,
        messages=[],
        max_round=num_nurses * 2,  # Each nurse gets a turn, followed by admin feedback
        speaker_selection_method="round_robin"  # Automatically rotate through speakers
    )
    
    manager = GroupChatManager(
        groupchat=group_chat, 
        llm_config={
            "config_list": [{
                "model": "gemini-2.5-flash", 
                "api_key": os.getenv("GOOGLE_API_KEY", "AIzaSyCv7CHW4E7atp6QW6WlySwLpV0l1HU1Ebc"),
                "api_type": "google"
            }], 
            "cache_seed": 42
        }
    )
    
    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    last_request_time: Optional[float] = None

    while not scenario.is_ended():
        # Start a new simulation tick
        # Reset for a new round of nurse-admin interactions
        alternator.reset()
        group_chat.messages = []  # Clear chat history for new tick
        
        if last_request_time is not None:
            elapsed = time.time() - last_request_time
            if elapsed < 30:
                time.sleep(30 - elapsed)
                
        # Initial state message for the tick
        state_msg = f"State:{scenario.get_state()}\nDiscuss actions."
        
        try:
            # Initialize chat with first nurse
            group_chat.next_speaker = nurse_agents[0]
            user_proxy.initiate_chat(manager, message=state_msg)
            last_request_time = time.time()
        except google_exceptions.ResourceExhausted as e:
            logger.warning(f"Rate limit hit")
            time.sleep(60)  # Wait longer on rate limits
            continue
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e).upper():
                logger.warning(f"Rate limit hit")
                retry_delay = 60
                time.sleep(retry_delay)
                continue
            logger.error(f"Unexpected error")
            time.sleep(30)  # Backoff
            continue
            
        # Process the messages we received
        processed_count = 0
        while processed_count < len(group_chat.messages):
            msg = group_chat.messages[processed_count]
            processed_count += 1
            
            # Only process nurse messages (skip admin messages we've already inserted)
            if msg.get("name", "").startswith("Nurse_"):
                nurse_name = msg.get("name")
                action = extract_action(msg.get("content", ""))
                
                # Apply action and get outcome
                outcome = None
                if action and "treat" in action:
                    outcome = scenario.apply_action(action)
                    if outcome:
                        scorer.update(outcome)
                        logger.info(f"Outcome for {action}: {outcome}")
                
                # Create admin response
                admin_response = create_admin_response(scenario, action, outcome, nurse_name)
                
                # Add admin response to chat history
                group_chat.messages.append({
                    "role": "assistant",
                    "name": "Admin",
                    "content": admin_response
                })
                processed_count += 1  # Count this admin message as processed
                
                # Set up for next nurse (if there are more nurses to respond)
                current_nurse_idx = int(nurse_name.split("_")[1]) - 1
                next_nurse_idx = (current_nurse_idx + 1) % num_nurses
                
                if next_nurse_idx > 0:  # Only continue if we haven't processed all nurses
                    # Add delay for rate limiting
                    if last_request_time is not None:
                        elapsed = time.time() - last_request_time
                        if elapsed < 30:
                            time.sleep(30 - elapsed)
                    
                    try:
                        # Set next speaker and continue chat
                        group_chat.next_speaker = nurse_agents[next_nurse_idx]
                        user_proxy.send(recipient=manager, message=f"Next speaker: {nurse_agents[next_nurse_idx].name}")
                        last_request_time = time.time()
                    except Exception as e:
                        logger.error(f"Error continuing chat: {e}")
                        time.sleep(30)
        
        # Advance simulation time after all nurses have acted
        scenario.tick()
        time.sleep(30)  # Additional throttle between ticks

    logger.info(f"Simulation ended. Final Score: {scorer}")
    logger.info(f"Detailed log saved to {log_path}")

if __name__ == "__main__":
    run_simulation(num_nurses=3, timeline_duration=30, supply_mode="normal")
