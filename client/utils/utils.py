from colorama import Fore, Style, init
from rich.console import Console
from rich.panel import Panel
from rich import box

console = Console()


def format_colored(timestamp, description):
    """Formats an entry with colors for emphasis."""
    return (
        f"{Fore.CYAN}--- Image Analysis ({timestamp}) ---\n"
        f"{Style.BRIGHT}{description}\n"
        f"{Style.RESET_ALL}" # Resets all formatting
    )

def print_framed_output(response_text):
    try:
        """
        Prints a given response text encapsulated within a rich.Panel.
        """
        current_time = datetime.now().strftime('%H:%M:%S')
        
        # Create a Panel for the output
        # - title: The timestamp for the panel's top border
        # - border_style: Color for the border (e.g., "bold blue", "green")
        # - box: Defines the style of the border (e.g., box.DOUBLE, box.ROUNDED)
        # - padding: Adds space around the text inside the panel
        panel = Panel(
            response_text,
            title=f"[bold green]Output at {current_time}[/bold green]",
            border_style="cyan",
            box=box.DOUBLE,  # Or box.ROUNDED, box.SQUARE, box.HEAVY
            padding=(1, 2) # (top/bottom, left/right)
        )
        
        # Print the panel to the console
        console.print(panel)
        console.print("\n") # Add an extra newline for more separation between panels
    except Exception as e:
        print(e)