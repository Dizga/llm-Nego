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
    table = Table(show_header=True, style="purple", header_style="bold magenta", padding=(1, 2))
    table.add_column("Role", width=12, justify="center")
    table.add_column("Content", justify="left")

    for message in conversation:
        if message["role"] == "context":
            style = "bright_red"
            role = 'Researcher'
        elif message["role"] == "assistant":
            style = "bright_yellow"
            role = 'Player'
        else:
            style = "bright_green"
            role = 'Intermediary'
        table.add_row(role, message["content"], style=style)

    console.print(table)

    # Export the table to HTML
    html = console.export_html()
    
    # Add custom CSS for the background color, text color, font size, and spacing
    custom_css = """
    <style>
        body, html {
            font-size: 24px;  /* Increased font size */
            background-color: #ffcccc !important;  /* Lighter red background */
            color: white !important;
            margin: 0;
            padding: 0;
        }
        .rich-terminal {
            background-color: #ffcccc !important;  /* Lighter red background */
            color: black !important;  /* Dark text color for contrast */
        }
        .rich-table {
            background-color: #ffcccc !important;  /* Lighter red background */
            color: black !important;
            padding: 15px;  /* Add spacing between contents */
            border-spacing: 15px;  /* Add spacing between cells */
        }
        .rich-table th, .rich-table td {
            padding: 10px 20px;  /* Padding within each cell */
        }
    </style>
    """
    html = html.replace("<head>", f"<head>{custom_css}")

    table_name = 'player_example.html'
    with open(table_name, "w") as f:
        f.write(html)
    print(f"HTML file '{table_name}' created successfully.")


