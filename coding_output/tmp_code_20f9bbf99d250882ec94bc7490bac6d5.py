import os

# --- Constants ---
# The name of the file where tasks will be stored.
TODO_FILE = "todo.txt"

# --- File Operations ---

def load_tasks() -> list[str]:
    """
    Loads tasks from the TODO_FILE.
    Each line in the file is considered a task.
    Returns a list of tasks. Returns an empty list if the file does not exist
    or if an error occurs during reading.
    """
    tasks = []
    if os.path.exists(TODO_FILE):
        try:
            with open(TODO_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    task = line.strip()
                    if task:  # Only add non-empty lines as tasks
                        tasks.append(task)
        except IOError as e:
            print(f"Error loading tasks: {e}")
            # In case of an error, we proceed with an empty list to prevent crashes
    return tasks

def save_tasks(tasks: list[str]):
    """
    Saves the current list of tasks to the TODO_FILE.
    Each task is written on a new line.
    """
    try:
        with open(TODO_FILE, 'w', encoding='utf-8') as f:
            for task in tasks:
                f.write(task + '\n')
    except IOError as e:
        print(f"Error saving tasks: {e}")

# --- Task Management Functions ---

def add_task(tasks: list[str], new_task: str):
    """
    Adds a new task to the list and then saves the updated list to the file.
    Ensures the task is not empty before adding.
    """
    if new_task: # Check if the task string is not empty
        tasks.append(new_task.strip())
        save_tasks(tasks)
        print(f"Task '{new_task}' added.")
    else:
        print("Task cannot be empty.")

def view_tasks(tasks: list[str]):
    """
    Displays all tasks currently in the list, numbered for easy reference.
    Provides a message if the list is empty.
    """
    if not tasks:
        print("\nYour to-do list is empty!")
    else:
        print("\n--- Your To-Do List ---")
        for i, task in enumerate(tasks, 1): # Start numbering from 1
            print(f"{i}. {task}")
        print("-----------------------")

def remove_task(tasks: list[str], task_index_str: str):
    """
    Removes a task from the list based on its 1-based index.
    Handles various invalid inputs (non-numeric, out-of-bounds index).
    After removal, the updated list is saved.
    """
    if not tasks:
        print("Your to-do list is empty. Nothing to remove.")
        return

    try:
        task_index = int(task_index_str)
        # Check if the provided index is within the valid range (1 to len(tasks))
        if 1 <= task_index <= len(tasks):
            # Use pop() to remove the task and get its value for feedback
            removed_task = tasks.pop(task_index - 1) # Adjust for 0-based list index
            save_tasks(tasks)
            print(f"Task '{removed_task}' removed.")
        else:
            print(f"Invalid task number. Please enter a number between 1 and {len(tasks)}.")
    except ValueError:
        # Catches errors if the input is not a valid integer
        print("Invalid input. Please enter a number.")
    except IndexError:
        # This catch is mostly for robustness; the `if 1 <= task_index` check should prevent it.
        print("Task number out of range.")

# --- User Interface ---

def display_menu():
    """
    Prints the main menu options to the console, guiding the user.
    """
    print("\n--- To-Do List Application ---")
    print("1. View tasks")
    print("2. Add task")
    print("3. Remove task")
    print("4. Exit")
    print("------------------------------")

def main():
    """
    Main function to run the to-do list application.
    It loads tasks, presents the menu, handles user input, and calls
    the appropriate task management functions in a loop until the user exits.
    """
    tasks = load_tasks() # Load tasks at the beginning of the application

    while True:
        display_menu()
        choice = input("Enter your choice: ").strip() # Get user input and remove whitespace

        if choice == '1':
            view_tasks(tasks)
        elif choice == '2':
            new_task = input("Enter the new task: ").strip()
            add_task(tasks, new_task)
        elif choice == '3':
            view_tasks(tasks) # Show tasks first so the user knows which one to remove
            if tasks: # Only prompt for removal if there are tasks to remove
                task_to_remove = input("Enter the number of the task to remove: ").strip()
                remove_task(tasks, task_to_remove)
            else:
                print("No tasks to remove.")
        elif choice == '4':
            print("Exiting To-Do List Application. Goodbye!")
            break # Exit the infinite loop
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

# --- Entry Point ---

if __name__ == "__main__":
    main()