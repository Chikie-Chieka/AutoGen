import os
import time
import json
import logging
import re
from datetime import datetime
import google.api_core.exceptions as google_exceptions
from autogen import GroupChat, GroupChatManager, UserProxyAgent

from modules.core import Scenario, Scorer
from modules.agents import setup_agents

def setup_logging() -> tuple[logging.Logger, str]:
    logger = logging.getLogger("triage_sim")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()

    log_filename = datetime.now().strftime("%H%M%S.%Y-%m-%d_log.txt")
    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), log_filename)

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

def extract_action(message: str) -> dict | None:
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
            else:
                return None  # Invalid structure
        except json.JSONDecodeError:
            pass
    return None

def run_simulation(num_nurses: int = 3, timeline_duration: int = 15, supply_mode: str = "normal"):
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
    manager = GroupChatManager(groupchat=group_chat, llm_config={"config_list": [{"model": "gemini-2.5-flash", "api_key": os.getenv("GOOGLE_API_KEY", "AIzaSyDSbQaFjrULmsKDVwE6w4qDS-rcrrZYoiA"), "api_type": "google", "generation_config": {"response_mime_type": "application/json"}}], "cache_seed": 42})

    last_request_time: float | None = None

    while not scenario.is_ended():
        counter = 0
        if last_request_time is not None:
            elapsed = time.time() - last_request_time
            if elapsed < 30:
                time.sleep(30 - elapsed)
        state_msg = f"State:{scenario.get_state()}\nDiscuss actions. Respond only with JSON as instructed."  # Concise, with reminder
        try:
            user_proxy.initiate_chat(manager, message=state_msg)
            last_request_time = time.time()
        except google_exceptions.ResourceExhausted as e:
            logger.warning(f"Rate limit hit (429). Waiting 60s. Error: {e}")
            time.sleep(60)
            continue
        except Exception as e:
            logger.error(f"Unexpected error during chat initiation: {e}")
            time.sleep(60)
            continue

        # Process actions from chat history
        treated_this_tick = set()  # To prevent overlaps
        for msg in group_chat.messages[-num_nurses:]:
            if "content" in msg and isinstance(msg["content"], str):
                action = extract_action(msg["content"])
                if action:
                    if "pass" in action:
                        logger.debug(f"{msg['name']} passed.")
                    else:
                        pid = scenario._normalize_patient_id(action['treat'])
                        if pid not in treated_this_tick:
                            outcome_dict = scenario.apply_action(action)
                            if outcome_dict:
                                nurse = msg.get("name", "Unknown")
                                scorer.update(outcome_dict, scenario.timeline, nurse)
                                logger.info(f"Outcome for {action}: {outcome_dict['outcome']}")
                                treated_this_tick.add(pid)
                        else:
                            logger.debug(f"Skipping duplicate treatment for P{pid} this tick")

        logger.info(f"State after treatments: {scenario.get_state()}")
        counter += 1
        if counter >= 2:
            group_chat.messages = []  # Clear history to reduce tokens every 2 chats
            counter = 0

        scenario.tick()
        for event_dict in scenario.get_last_events():
            scorer.update(event_dict, scenario.timeline)
        time.sleep(30)  # Increased throttle to 30s to avoid 429 errors (was 15s)

    logger.info(f"Simulation ended. Final Score: {scorer}")
    logger.info(f"Detailed log saved to {log_path}")