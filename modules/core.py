import random
import logging
from typing import List, Dict

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
                    effect = action.get('effect', -2)  # Improve by 2
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