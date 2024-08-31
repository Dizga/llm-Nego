import json
from rich.console import Console
from rich.table import Table

def showcase(file_or_string):
    if file_or_string.endswith('.json'):
        # Load JSON from file
        file_or_string = '../' + file_or_string
        with open(file_or_string, 'r') as f:
            conversation = json.load(f)
    else:
        # Load JSON from string
        conversation = json.loads(file_or_string)

    # Initialize the console
    console = Console(record=True)

    # Create a table to display the conversation
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Role", style="dim", width=12)
    table.add_column("Content")

    for message in conversation:
        style = "blue" if message["role"] == "assistant" else "green"
        role = 'Player' if message["role"] == "assistant" else 'Moderator'
        table.add_row(role, message["content"], style=style)

    console.print(table)

    # Export the table to HTML
    html = console.export_html()
    custom_css = """
    <style>
        body {
            font-size: 12px;
        }
    </style>
    """
    html = html.replace("<head>", f"<head>{custom_css}")

    table_name = 'player_example.html'
    with open(table_name, "w") as f:
        f.write(html)
    print(f"HTML file '{table_name}' created successfully.")