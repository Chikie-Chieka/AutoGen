import os
from typing import List
from autogen import AssistantAgent

def setup_agents(num_nurses: int = 3) -> List[AssistantAgent]:
    """Create nurse agents with Gemini model."""
    # Load config for Gemini with caching and forced JSON output
    config_list = [
        {
            "model": "gemini-2.5-flash",
            "api_key": os.getenv("GOOGLE_API_KEY", "AIzaSyDSbQaFjrULmsKDVwE6w4qDS-rcrrZYoiA"),
            "api_type": "google",
            "generation_config": {
                "response_mime_type": "application/json"
            }
        }
    ]
    llm_config = {"config_list": config_list, "cache_seed": 42}  # Enable caching

    system_message = """You are a nurse, your task is to provide treatment for patient, each treatment will lessen the severity of a patient condition by 2, 
    prioritize highest severity. Coordinate with other nurses by reading prior messages in this chat; e.g., if another nurse is treating a patient this tick, choose 
    a different one to avoid overlaps. You have 1 action per tick. Only pass if there is no patient who need treatment or when there are no patient left to treat.
    Output ONLY one valid JSON like {"treat": "P2", "with": "meds", "effect": -2} or {"pass": true}, you may explain your action in 1 short sentence. No extra text, 
    no explanations, no code blocks."""

    agents = []
    for i in range(1, num_nurses + 1):
        agent = AssistantAgent(
            name=f"Nurse_{i}",
            system_message=system_message,
            llm_config=llm_config
        )
        agents.append(agent)

    return agents