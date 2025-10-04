from modules.simulation import run_simulation

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
    # Prompt user to choose simulation mode
    mode, event_description = get_simulation_mode()
    
    # Run simulation with appropriate settings based on mode
    run_simulation(
        num_nurses=3, 
        timeline_duration=15, 
        supply_mode="normal", 
        event_description=event_description
    )