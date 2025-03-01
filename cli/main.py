import os
import click
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from moghedien.bloodhound.parser import BloodHoundParser
from moghedien.core.graph import ADGraph
from moghedien.core.pathfinder import AttackPathFinder

console = Console()

@click.group()
def cli():
    """Moghedien: AI-driven attack path analysis for Active Directory."""
    pass

@cli.command()
@click.option('--data-dir', '-d', required=True, help='Directory containing BloodHound JSON files')
def analyze(data_dir):
    """Analyze BloodHound data to identify attack paths."""
    if not os.path.isdir(data_dir):
        console.print(f"[bold red]Error:[/bold red] {data_dir} is not a valid directory")
        return
        
    try:
        console.print("[bold blue]Loading BloodHound data...[/bold blue]")
        parser = BloodHoundParser()
        parser.parse_directory(data_dir)
        
        console.print("[bold blue]Building graph representation...[/bold blue]")
        graph = ADGraph()
        graph.build_from_bloodhound(parser)
        
        console.print("[bold blue]Ready to analyze attack paths.[/bold blue]")
        interactive_mode(graph)
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

def interactive_mode(graph):
    """Interactive mode for attack path analysis."""
    pathfinder = AttackPathFinder(graph)
    
    while True:
        console.print("\n[bold cyan]Moghedien Interactive Mode[/bold cyan]")
        console.print("1. Find paths to Domain Admin")
        console.print("2. Find paths to high-value targets")
        console.print("3. Search for a specific object")
        console.print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            source = input("Enter source object name or SID: ")
            source_id = find_object_id(graph, source)
            
            if source_id:
                console.print(f"[bold green]Finding paths from {source} to Domain Admin...[/bold green]")
                paths = pathfinder.find_paths_to_domain_admin(source_id)
                display_paths(paths)
            else:
                console.print("[bold red]Source object not found[/bold red]")
                
        elif choice == '2':
            source = input("Enter source object name or SID: ")
            source_id = find_object_id(graph, source)
            
            if source_id:
                console.print(f"[bold green]Finding paths from {source} to high-value targets...[/bold green]")
                paths = pathfinder.find_paths_to_high_value_targets(source_id)
                display_paths(paths)
            else:
                console.print("[bold red]Source object not found[/bold red]")
                
        elif choice == '3':
            search_term = input("Enter search term: ")
            search_objects(graph, search_term)
            
        elif choice == '4':
            console.print("[bold green]Exiting Moghedien.[/bold green]")
            break
            
        else:
            console.print("[bold red]Invalid choice. Please try again.[/bold red]")

def find_object_id(graph, name_or_id):
    """Find an object ID by name or ID."""
    # Check if it's already an ID in the graph
    if graph.graph.has_node(name_or_id):
        return name_or_id
        
    # Search by name (case-insensitive partial match)
    name_lower = name_or_id.lower()
    for node_id, attrs in graph.graph.nodes(data=True):
        node_name = attrs.get('name', '').lower()
        if name_lower in node_name:
            return node_id
            
    return None

def search_objects(graph, search_term):
    """Search for objects in the graph."""
    search_term = search_term.lower()
    results = []
    
    for node_id, attrs in graph.graph.nodes(data=True):
        node_name = attrs.get('name', '').lower()
        if search_term in node_name or search_term in node_id.lower():
            results.append((node_id, attrs))
            
    if not results:
        console.print("[bold yellow]No matching objects found.[/bold yellow]")
        return
        
    table = Table(title=f"Search Results for '{search_term}'")
    table.add_column("ID", style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")
    
    for node_id, attrs in results:
        table.add_row(
            node_id,
            attrs.get('name', ''),
            attrs.get('type', 'Unknown')
        )
        
    console.print(table)

def display_paths(paths):
    """Display attack paths in a readable format."""
    if not paths:
        console.print("[bold yellow]No paths found.[/bold yellow]")
        return
        
    console.print(f"[bold green]Found {len(paths)} potential attack paths.[/bold green]")
    
    for i, path_data in enumerate(paths):
        console.print(f"\n[bold cyan]Path {i+1}[/bold cyan] (Score: {path_data['total_score']:.2f})")
        console.print(f"Success Probability: {path_data['success_score']:.2f}%")
        console.print(f"Stealth Score: {path_data['stealth_score']:.2f}%")
        console.print(f"Path Length: {path_data['length']} steps")
        
        steps_table = Table(show_header=True, header_style="bold magenta")
        steps_table.add_column("Step")
        steps_table.add_column("Source")
        steps_table.add_column("Technique")
        steps_table.add_column("Target")
        
        for j, step in enumerate(path_data['steps']):
            steps_table.add_row(
                str(j+1),
                step['source'],
                step['relationship'],
                step['target']
            )
            
        console.print(steps_table)
        
        # Display attack techniques for the path
        console.print("[bold green]Recommended Attack Techniques:[/bold green]")
        
        for j, step in enumerate(path_data['steps']):
            console.print(f"\n[bold]Step {j+1}: {step['source']} -> {step['target']}[/bold]")
            
            techniques = step.get('techniques', [])
            for technique in techniques:
                panel = Panel(
                    f"[bold]{technique['name']}[/bold]\n\n"
                    f"{technique['description']}\n\n"
                    f"[dim]{technique['command']}[/dim]",
                    title=f"Technique for {step['relationship']}",
                    border_style="green"
                )
                console.print(panel)

if __name__ == '__main__':
    cli()
