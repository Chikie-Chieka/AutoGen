import os
from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# AutoGen configuration for Gemini
config_list = [
    {
        "model": "gemini-2.5-flash",
        "api_key": gemini_api_key,
        "api_type": "google",
        "base_url": "https://generativelanguage.googleapis.com/v1beta"
    }
]

# Initialize agents
coder = AssistantAgent(
    name="Coder",
    llm_config={"config_list": config_list},
    system_message="You are a skilled programmer. Write clean, functional Python code for a command-line to-do list app that allows adding, viewing, and removing tasks. Ensure the code is modular and well-commented."
)

reviewer = AssistantAgent(
    name="Reviewer",
    llm_config={"config_list": config_list},
    system_message="You are a code reviewer and tester. Review the code for errors, suggest improvements, and propose test cases. If the code is flawed, provide specific fixes. You can execute the code to verify it works."
)

# User proxy to initiate the task
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "coding_output",
        "use_docker": False
    }
)

# Create output directory if it doesn't exist
os.makedirs("coding_output", exist_ok=True)

# Start the collaboration
user_proxy.initiate_chat(
    coder,
    message="Create a simple command-line to-do list application in Python. It should allow users to add tasks, view the task list, and remove tasks. The Reviewer should check the code and suggest improvements or fixes."
)