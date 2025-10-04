# This is a sample testing program using AutoGen to set up two agents with different roles,
# both powered by the Gemini 2.5 Flash model. The agents will collaborate on a simple task:
# writing and testing a Python function to calculate the factorial of a number.

# Prerequisites:
# 1. Install required packages:
#    pip install pyautogen google-generativeai
# 2. Obtain a Google API key from https://aistudio.google.com/ and replace 'YOUR_GOOGLE_API_KEY' below.
# 3. Run this script with Python 3.

import autogen
from autogen import AssistantAgent, UserProxyAgent

# Configuration for the Gemini model
config_list = [
    {
        "model": "gemini-2.5-pro",
        "api_key": "AIzaSyCv7CHW4E7atp6QW6WlySwLpV0l1HU1Ebc",  # Replace with your actual API key
        "api_type": "google"
    }
]

# Create the Coder agent: Its role is to write code based on requirements.
coder = AssistantAgent(
    name="Coder",
    system_message="You are a helpful AI coder. Your role is to write Python code to solve the given task. Do not execute code; just provide it.",
    llm_config={"config_list": config_list},
)

# Create the Tester agent: Its role is to test and verify the code provided by the Coder.
tester = AssistantAgent(
    name="Tester",
    system_message="You are a meticulous AI tester. Your role is to review the code, suggest improvements if needed, and test it by writing test cases or simulating execution. Provide feedback.",
    llm_config={"config_list": config_list},
)

# Create a UserProxyAgent to initiate the conversation (simulates a user or coordinator).
# Set human_input_mode to "NEVER" for fully autonomous run; change to "ALWAYS" if you want manual input.
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,  # No auto replies from user proxy
    code_execution_config=False,  # Disable code execution for safety in this example
)

# Define a simple task for testing
task = """
Write a Python function to calculate the factorial of a number n (where n >= 0).
Then, test it with inputs: 0, 1, 5, and handle a negative input case.
"""

# Initiate the chat: User proxy starts by sending the task to the Coder.
# The agents will converse back and forth.
user_proxy.initiate_chat(
    coder,
    message=task,
    # You can chain more agents if needed, but here we start with Coder, and Tester can be involved manually or in a group chat.
)

# For a group chat involving both agents:
groupchat = autogen.GroupChat(agents=[coder, tester], messages=[], max_round=5)  # Limit to 5 rounds for testing
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

# Start the group chat with the task
user_proxy.initiate_chat(
    manager,
    message=task
)

# The conversation history will be printed in the console as the agents interact.