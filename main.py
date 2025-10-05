import os
from modules.simulation import run_simulation
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
def check_api_key():
    """Check if OpenRouter API key is set and provide instructions if not."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    print(api_key) ###TODO: not taking .env file
    if not api_key:
        print("\n⚠️ WARNING: OPENROUTER_API_KEY environment variable is not set.")
        print("To use this application, you need to set up an OpenRouter API key:")
        print("1. Sign up at https://openrouter.ai/")
        print("2. Create an API key")
        print("3. Set the environment variable with:")
        print("   - Windows: set OPENROUTER_API_KEY=your_api_key_here")
        print("   - Linux/MacOS: export OPENROUTER_API_KEY=your_api_key_here")
        print("\nDefault fallback key is being used, but it may have usage limitations.")
        input("Press Enter to continue...")

def get_user_event_description():
    """Prompt user for event description and return it as a string."""
    print("\n=== Manual Event Entry ===")
    print("Enter your event descriptions, one per line.")
    print("Supported formats:")
    print("- 'At tick 5, new patient P6 with severity 7 arrives'")
    print("- 'At tick 10, emergency for P3'")
    print("Type 'END' on a new line when you're finished:")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":   
            break
        lines.append(line)
    return "\n".join(lines)

def get_simulation_mode():
    """Prompt user to choose between Randomised and Manual simulation modes."""
    print("\n=== Hospital Triage Simulation ===")
    print("Choose simulation mode:")
    print("1. Randomised - Events are generated automatically")
    print("2. Manual - Specify your own event timeline")
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == "1":
            return "randomised", None
        elif choice == "2":
            return "manual", get_user_event_description()
        else:
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    # Check if API key is properly set
    check_api_key()
    
    print("\n=== Hospital Triage Simulation (OpenRouter + Gemini) ===")
    print("This simulation uses Google's Gemini model via OpenRouter API")
    
    # Prompt user to choose simulation mode
    mode, event_description = get_simulation_mode()
    
    # Run simulation with appropriate settings based on mode
    run_simulation(
        num_nurses=3, 
        timeline_duration=15, 
        supply_mode="normal", 
        event_description=event_description
    )