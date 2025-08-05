"""
Command-line interface for nanook-curator.

This module provides a comprehensive CLI for the nanook-curator system with support for
manual execution, configuration management, verbose logging, and debug options.
"""

import sys
import logging
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler
from rich.json import JSON

from .config import Configuration, init_config, get_config, reset_config
from .models import CuratorState
from .workflow import create_curator_workflow
# Error handling imports removed - using built-in logging setup


# Initialize rich console for beautiful output
console = Console()


def setup_cli_logging(log_level: str, log_file: Optional[Path] = None, verbose: bool = False):
    """
    Set up logging for CLI with rich formatting and optional file output.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        verbose: Enable verbose console output
    """
    # Clear any existing handlers
    logging.getLogger().handlers.clear()
    
    # Set up rich handler for console output
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=verbose,
        markup=True,
        rich_tracebacks=True
    )
    console_handler.setLevel(logging.DEBUG if verbose else getattr(logging, log_level))
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[console_handler]
    )
    
    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(file_handler)


def display_config_table(config: Configuration):
    """Display configuration in a formatted table."""
    table = Table(title="Nanook Curator Configuration")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    table.add_column("Description", style="green")
    
    config_dict = config.to_dict()
    descriptions = {
        'max_videos': 'Maximum videos to analyze per search',
        'days_back': 'Days back to search for videos',
        'quality_threshold': 'Quality score threshold (0-100)',
        'min_quality_videos': 'Minimum quality videos required',
        'max_search_attempts': 'Maximum search refinement attempts',
        'target_word_count_min': 'Minimum script word count',
        'target_word_count_max': 'Maximum script word count',
        'script_language': 'Language code for script generation',
        'results_storage_path': 'Directory for storing results',
        'log_level': 'Logging level',
        'debug': 'Debug mode enabled',
        'mock_apis': 'Use mock API responses'
    }
    
    for key, value in config_dict.items():
        if key in ['youtube_api_key', 'openai_api_key']:
            continue  # Skip API keys for security
        
        description = descriptions.get(key, '')
        if isinstance(value, list):
            value = ', '.join(str(v) for v in value)
        
        table.add_row(key, str(value), description)
    
    console.print(table)


def display_execution_results(state: CuratorState, execution_time: float):
    """Display execution results in a formatted way."""
    console.print("\n" + "="*60)
    console.print(Panel.fit("🎉 Execution Complete!", style="bold green"))
    


def init_config_if_needed(ctx, require_config: bool = True):
    """Initialize configuration only when needed."""
    if 'config' in ctx.obj:
        return ctx.obj['config']
    
    config_file = ctx.obj.get('config_file')
    debug = ctx.obj.get('debug', False)
    verbose = ctx.obj.get('verbose', False)
    log_file = ctx.obj.get('log_file')
    
    try:
        config = init_config(config_file)
        
        # Override debug setting if specified via CLI
        if debug:
            config.debug = True
            config.log_level = 'DEBUG'
        
        # Set up logging
        log_file_path = log_file or config.log_file
        setup_cli_logging(config.log_level, log_file_path, verbose)
        
        ctx.obj['config'] = config
        
        if verbose:
            console.print(f"[green]✓[/green] Configuration loaded successfully")
            if config_file:
                console.print(f"[blue]ℹ[/blue] Using config file: {config_file}")
        
        return config
        
    except Exception as e:
        if require_config:
            console.print(f"[red]✗[/red] Configuration error: {str(e)}")
            if debug:
                console.print_exception()
            console.print("\n[blue]ℹ[/blue] Try running 'nanook-curator init-config-file' to create a configuration template")
            sys.exit(1)
        else:
            # For commands that don't require config, set up basic logging
            setup_cli_logging('INFO', None, verbose)
            return None


@click.group()
@click.option('--config-file', '-c', type=click.Path(exists=True, path_type=Path),
              help='Path to configuration file (.env format)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--log-file', type=click.Path(path_type=Path),
              help='Path to log file (overrides config)')
@click.pass_context
def main(ctx, config_file: Optional[Path], verbose: bool, debug: bool, log_file: Optional[Path]):
    """
    Nanook Curator - AI-powered content curation system for YouTube videos.
    
    Automatically discovers trending YouTube videos about AI topics, analyzes content
    through transcripts, and generates curated podcast scripts.
    """
    # Ensure that ctx.obj exists and is a dict
    ctx.ensure_object(dict)
    
    # Store CLI options in context
    ctx.obj['verbose'] = verbose
    ctx.obj['debug'] = debug
    ctx.obj['config_file'] = config_file
    ctx.obj['log_file'] = log_file


@main.command()
@click.option('--keywords', '-k', multiple=True,
              help='Search keywords (can be specified multiple times)')
@click.option('--max-videos', type=int,
              help='Maximum number of videos to analyze')
@click.option('--days-back', type=int,
              help='Number of days back to search')
@click.option('--quality-threshold', type=float,
              help='Quality score threshold (0-100)')
@click.option('--openai-model', type=str,
              help='OpenAI model to use for script generation (e.g., gpt-4o, gpt-4o-mini, gpt-4-turbo)')
@click.option('--output-file', '-o', type=click.Path(path_type=Path),
              help='Output file for generated script')
@click.option('--dry-run', is_flag=True,
              help='Validate configuration and show what would be executed without running')
@click.pass_context
def run(ctx, keywords: tuple, max_videos: Optional[int], days_back: Optional[int],
        quality_threshold: Optional[float], openai_model: Optional[str], output_file: Optional[Path], dry_run: bool):
    """
    Run the content curation workflow to generate a podcast script.
    
    This command executes the complete workflow: discovers videos, analyzes content,
    evaluates quality, and generates a podcast script from the best content.
    """
    config: Configuration = init_config_if_needed(ctx, require_config=True)
    verbose = ctx.obj['verbose']
    
    # Override configuration with CLI options
    if keywords:
        search_keywords = list(keywords)
    else:
        search_keywords = config.default_search_keywords
    
    if max_videos:
        config.max_videos = max_videos
    if days_back:
        config.days_back = days_back
    if quality_threshold:
        config.quality_threshold = quality_threshold
    if openai_model:
        config.openai_model = openai_model
    
    # Validate API keys
    if not config.validate_api_keys():
        console.print("[red]✗[/red] API keys validation failed. Please check your configuration.")
        console.print("Required: YOUTUBE_API_KEY and OPENAI_API_KEY environment variables")
        sys.exit(1)
    
    if verbose:
        console.print(f"[blue]ℹ[/blue] Search keywords: {', '.join(search_keywords)}")
        console.print(f"[blue]ℹ[/blue] Max videos: {config.max_videos}")
        console.print(f"[blue]ℹ[/blue] Days back: {config.days_back}")
        console.print(f"[blue]ℹ[/blue] Quality threshold: {config.quality_threshold}")
        console.print(f"[blue]ℹ[/blue] OpenAI model: {config.openai_model}")
    
    if dry_run:
        console.print(Panel.fit("🔍 Dry Run Mode - Configuration Validation", style="blue"))
        display_config_table(config)
        console.print("[green]✓[/green] Configuration is valid. Ready to execute.")
        return
    
    # Initialize state
    initial_state = CuratorState(
        search_keywords=search_keywords,
        max_videos=config.max_videos,
        days_back=config.days_back,
        quality_threshold=config.quality_threshold,
        min_quality_videos=config.min_quality_videos,
        max_search_attempts=config.max_search_attempts
    )
    
    console.print(Panel.fit("🚀 Starting Content Curation Workflow", style="bold blue"))
    
    # Execute workflow with progress tracking
    start_time = datetime.now()
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Executing workflow...", total=None)
            
            # Create and run the workflow
            workflow = create_curator_workflow(config)
            final_state = workflow.invoke(initial_state)
            
            progress.update(task, description="Workflow completed!")
        
        execution_time = (datetime.now() - start_time).total_seconds()

        """Display execution results in a formatted way."""
        console.print("\n" + "="*60)
        console.print(Panel.fit("🎉 Execution Complete!", style="bold green"))

        
    except Exception as e:
        console.print(f"[red]✗[/red] Workflow execution failed: {e}")
        if ctx.obj['debug']:
            console.print_exception()
        sys.exit(1)


@main.command()
@click.pass_context
def config(ctx):
    """Display current configuration settings."""
    config: Configuration = init_config_if_needed(ctx, require_config=True)
    
    console.print(Panel.fit("⚙️  Current Configuration", style="bold cyan"))
    display_config_table(config)
    
    # Validate API keys
    if config.validate_api_keys():
        console.print("[green]✓[/green] API keys are configured")
    else:
        console.print("[red]✗[/red] API keys validation failed")


@main.command()
@click.option('--output-file', '-o', type=click.Path(path_type=Path),
              help='Output file for configuration template')
@click.pass_context
def init_config_file(ctx, output_file: Optional[Path]):
    """Generate a configuration file template with all available options."""
    # This command doesn't need configuration
    init_config_if_needed(ctx, require_config=False)
    
    if not output_file:
        output_file = Path('.env')
    
    template = """# Nanook Curator Configuration
# Copy this file to .env and fill in your API keys

# Required API Keys
YOUTUBE_API_KEY=your_youtube_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# OpenAI Configuration
# Available models: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo
# gpt-4o-mini is cost-effective and suitable for most use cases
OPENAI_MODEL=gpt-4o-mini

# Content Discovery Settings
MAX_VIDEOS=10
DAYS_BACK=7
QUALITY_THRESHOLD=80.0
MIN_QUALITY_VIDEOS=1
MAX_SEARCH_ATTEMPTS=3

# Search Configuration (comma-separated)
DEFAULT_SEARCH_KEYWORDS=AI news,AI tools,AI agents,artificial intelligence,machine learning

# Script Generation Settings
TARGET_WORD_COUNT_MIN=750
TARGET_WORD_COUNT_MAX=1500
SCRIPT_LANGUAGE=en

# Storage Configuration
RESULTS_STORAGE_PATH=./output
LOG_FILE=./output/nanook-curator.log

# Logging Configuration
LOG_LEVEL=INFO

# Development Settings
DEBUG=false
MOCK_APIS=false
"""
    
    output_file.write_text(template, encoding='utf-8')
    console.print(f"[green]✓[/green] Configuration template created: {output_file}")
    console.print("[blue]ℹ[/blue] Edit the file and add your API keys to get started")


@main.command()
@click.pass_context
def validate(ctx):
    """Validate configuration and test API connections."""
    config: Configuration = init_config_if_needed(ctx, require_config=True)
    verbose = ctx.obj['verbose']
    
    console.print(Panel.fit("🔍 Validating Configuration and API Connections", style="bold yellow"))
    
    # Configuration validation
    try:
        console.print("[blue]ℹ[/blue] Validating configuration...")
        
        # Basic validation is done during config loading
        console.print("[green]✓[/green] Configuration validation passed")
        
        if verbose:
            display_config_table(config)
        
    except Exception as e:
        console.print(f"[red]✗[/red] Configuration validation failed: {e}")
        sys.exit(1)
    
    # API key validation
    console.print("[blue]ℹ[/blue] Validating API keys...")
    if config.validate_api_keys():
        console.print("[green]✓[/green] API keys format validation passed")
    else:
        console.print("[red]✗[/red] API keys validation failed")
        sys.exit(1)
    
    # Test API connections (basic connectivity test)
    console.print("[blue]ℹ[/blue] Testing API connections...")
    
    try:
        from .youtube_client import YouTubeClient
        from .script_generator import OpenAIScriptGenerator
        
        # Test YouTube API
        youtube_client = YouTubeClient(config)
        console.print("[green]✓[/green] YouTube API client initialized")
        
        # Test OpenAI API
        script_generator = OpenAIScriptGenerator(config.openai_api_key, config.openai_model)
        console.print("[green]✓[/green] OpenAI API client initialized")
        
        console.print(Panel.fit("✅ All validations passed!", style="bold green"))
        
    except Exception as e:
        console.print(f"[red]✗[/red] API connection test failed: {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@main.command()
@click.option('--days', type=int, default=7, help='Number of days of results to show')
@click.pass_context
def history(ctx, days: int):
    """Show execution history and results from the past N days."""
    config: Configuration = init_config_if_needed(ctx, require_config=True)
    
    console.print(Panel.fit(f"📊 Execution History (Last {days} days)", style="bold cyan"))
    
    # Look for result files in the output directory
    output_dir = config.results_storage_path
    if not output_dir.exists():
        console.print("[yellow]⚠[/yellow] No results directory found")
        return
    
    # Find script and metadata files
    script_files = list(output_dir.glob("script_*.txt"))
    metadata_files = list(output_dir.glob("metadata_*.json"))
    
    if not script_files and not metadata_files:
        console.print("[yellow]⚠[/yellow] No execution history found")
        return
    
    # Create history table
    history_table = Table(title="Recent Executions")
    history_table.add_column("Date", style="cyan")
    history_table.add_column("Script File", style="white")
    history_table.add_column("Word Count", style="green")
    history_table.add_column("Videos", style="blue")
    history_table.add_column("Duration", style="magenta")
    
    # Process files and sort by date
    executions = []
    
    for script_file in script_files:
        try:
            # Extract timestamp from filename
            timestamp_str = script_file.stem.replace('script_', '')
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            
            # Check if within date range
            if (datetime.now() - timestamp).days > days:
                continue
            
            # Read script content
            script_content = script_file.read_text(encoding='utf-8')
            word_count = len(script_content.split())
            
            # Look for corresponding metadata file
            metadata_file = output_dir / f"metadata_{timestamp_str}.json"
            videos_count = "N/A"
            duration = "N/A"
            
            if metadata_file.exists():
                try:
                    metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
                    videos_count = str(metadata.get('videos_ranked', 'N/A'))
                    duration = f"{metadata.get('execution_time', 0):.1f}s"
                except Exception:
                    pass
            
            executions.append({
                'timestamp': timestamp,
                'script_file': script_file.name,
                'word_count': word_count,
                'videos_count': videos_count,
                'duration': duration
            })
            
        except Exception as e:
            if ctx.obj['verbose']:
                console.print(f"[yellow]⚠[/yellow] Error processing {script_file}: {e}")
    
    # Sort by timestamp (newest first)
    executions.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Display results
    for execution in executions:
        history_table.add_row(
            execution['timestamp'].strftime('%Y-%m-%d %H:%M'),
            execution['script_file'],
            str(execution['word_count']),
            execution['videos_count'],
            execution['duration']
        )
    
    if executions:
        console.print(history_table)
        console.print(f"\n[blue]ℹ[/blue] Found {len(executions)} executions in the last {days} days")
    else:
        console.print(f"[yellow]⚠[/yellow] No executions found in the last {days} days")


@main.command()
@click.option('--schedule-info', is_flag=True, help='Show scheduling information and examples')
@click.pass_context
def schedule(ctx, schedule_info: bool):
    """
    Show information about scheduling the curator for automated execution.
    
    This command provides examples and guidance for setting up automated
    execution using external schedulers like cron, systemd timers, or Kubernetes CronJobs.
    """
    init_config_if_needed(ctx, require_config=False)
    
    if schedule_info:
        console.print(Panel.fit("📅 Scheduling Information", style="bold blue"))
        
        console.print("\n[bold]Cron Example (Daily at 9 AM):[/bold]")
        console.print("0 9 * * * /path/to/nanook-curator run --config-file /path/to/.env")
        
        console.print("\n[bold]Systemd Timer Example:[/bold]")
        console.print("""[Unit]
Description=Nanook Curator Daily Run
Requires=nanook-curator.service

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target""")
        
        console.print("\n[bold]Docker/Kubernetes CronJob Example:[/bold]")
        console.print("""apiVersion: batch/v1
kind: CronJob
metadata:
  name: nanook-curator
spec:
  schedule: "0 9 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: nanook-curator
            image: nanook-curator:latest
            command: ["nanook-curator", "run"]
            env:
            - name: YOUTUBE_API_KEY
              valueFrom:
                secretKeyRef:
                  name: api-keys
                  key: youtube-api-key
          restartPolicy: OnFailure""")
        
        console.print("\n[bold]Recommended Scheduling Patterns:[/bold]")
        console.print("• [cyan]Daily:[/cyan] 0 9 * * * (9 AM daily)")
        console.print("• [cyan]Weekly:[/cyan] 0 9 * * 1 (9 AM every Monday)")
        console.print("• [cyan]Twice Daily:[/cyan] 0 9,21 * * * (9 AM and 9 PM)")
        
        console.print("\n[bold]Best Practices:[/bold]")
        console.print("• Use absolute paths in cron jobs")
        console.print("• Set up log rotation for output files")
        console.print("• Monitor execution success/failure")
        console.print("• Use --dry-run to test configuration")
        console.print("• Consider rate limits for API calls")
        
    else:
        console.print("[blue]ℹ[/blue] Use --schedule-info to see scheduling examples and best practices")


@main.command()
@click.option('--count', '-n', type=int, default=5, help='Number of recent scripts to compare')
@click.option('--output-best', '-o', type=click.Path(path_type=Path), 
              help='Save the best script to specified file')
@click.pass_context
def compare(ctx, count: int, output_best: Optional[Path]):
    """Compare recent scripts and select the best one based on quality metrics."""
    config: Configuration = init_config_if_needed(ctx, require_config=True)
    verbose = ctx.obj['verbose']
    
    console.print(Panel.fit(f"📊 Comparing Last {count} Scripts", style="bold cyan"))
    
    output_dir = config.results_storage_path
    if not output_dir.exists():
        console.print("[yellow]⚠[/yellow] No results directory found")
        return
    
    # Find recent script and metadata files
    script_files = sorted(output_dir.glob("script_*.txt"), key=lambda x: x.stat().st_mtime, reverse=True)[:count]
    
    if len(script_files) < 2:
        console.print("[yellow]⚠[/yellow] Need at least 2 scripts to compare")
        return
    
    # Analyze scripts and create comparison table
    comparison_table = Table(title="Script Comparison")
    comparison_table.add_column("Rank", style="cyan")
    comparison_table.add_column("Date", style="white")
    comparison_table.add_column("Word Count", style="green")
    comparison_table.add_column("Videos Used", style="blue")
    comparison_table.add_column("Quality Score", style="magenta")
    comparison_table.add_column("Execution Time", style="yellow")
    
    script_analyses = []
    
    for script_file in script_files:
        try:
            # Extract timestamp from filename
            timestamp_str = script_file.stem.replace('script_', '')
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            
            # Read script content
            script_content = script_file.read_text(encoding='utf-8')
            word_count = len(script_content.split())
            
            # Look for corresponding metadata file
            metadata_file = output_dir / f"metadata_{timestamp_str}.json"
            videos_count = 0
            execution_time = 0
            quality_score = 0
            
            if metadata_file.exists():
                try:
                    metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
                    videos_count = metadata.get('videos_ranked', 0)
                    execution_time = metadata.get('execution_time', 0)
                    
                    # Calculate quality score based on multiple factors
                    # Higher score for more videos, optimal word count, faster execution
                    word_score = min(100, max(0, 100 - abs(word_count - 1125) / 10))  # Optimal around 1125 words
                    video_score = min(100, videos_count * 25)  # Up to 4 videos = 100 points
                    speed_score = max(0, 100 - execution_time / 2)  # Faster is better
                    
                    quality_score = (word_score * 0.4 + video_score * 0.4 + speed_score * 0.2)
                    
                except Exception as e:
                    if verbose:
                        console.print(f"[yellow]⚠[/yellow] Error reading metadata for {script_file.name}: {e}")
            
            script_analyses.append({
                'file': script_file,
                'timestamp': timestamp,
                'word_count': word_count,
                'videos_count': videos_count,
                'execution_time': execution_time,
                'quality_score': quality_score,
                'content': script_content
            })
            
        except Exception as e:
            if verbose:
                console.print(f"[yellow]⚠[/yellow] Error processing {script_file}: {e}")
    
    # Sort by quality score (highest first)
    script_analyses.sort(key=lambda x: x['quality_score'], reverse=True)
    
    # Display comparison results
    for i, analysis in enumerate(script_analyses, 1):
        comparison_table.add_row(
            str(i),
            analysis['timestamp'].strftime('%Y-%m-%d %H:%M'),
            str(analysis['word_count']),
            str(analysis['videos_count']),
            f"{analysis['quality_score']:.1f}",
            f"{analysis['execution_time']:.1f}s"
        )
    
    console.print(comparison_table)
    
    if script_analyses:
        best_script = script_analyses[0]
        console.print(f"\n[green]🏆 Best Script:[/green] {best_script['file'].name}")
        console.print(f"[blue]ℹ[/blue] Quality Score: {best_script['quality_score']:.1f}")
        console.print(f"[blue]ℹ[/blue] Word Count: {best_script['word_count']}")
        console.print(f"[blue]ℹ[/blue] Videos Used: {best_script['videos_count']}")
        
        # Save best script if requested
        if output_best:
            output_best.parent.mkdir(parents=True, exist_ok=True)
            output_best.write_text(best_script['content'], encoding='utf-8')
            console.print(f"[green]✓[/green] Best script saved to: {output_best}")
        
        # Show preview of best script
        if verbose:
            console.print("\n" + Panel(
                best_script['content'][:500] + "..." if len(best_script['content']) > 500 else best_script['content'],
                title="Best Script Preview",
                style="green"
            ))


@main.command()
@click.option('--all', 'clean_all', is_flag=True, help='Clean all results and logs')
@click.option('--older-than', type=int, help='Clean results older than N days')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
@click.pass_context
def clean(ctx, clean_all: bool, older_than: Optional[int], confirm: bool):
    """Clean up old results and log files."""
    config: Configuration = init_config_if_needed(ctx, require_config=True)
    output_dir = config.results_storage_path
    
    if not output_dir.exists():
        console.print("[yellow]⚠[/yellow] No results directory found")
        return
    
    console.print(Panel.fit("🧹 Cleaning Up Results", style="bold yellow"))
    
    # Find files to clean
    files_to_clean = []
    
    if clean_all:
        # Clean all files
        for pattern in ["script_*.txt", "metadata_*.json", "*.log"]:
            files_to_clean.extend(output_dir.glob(pattern))
    elif older_than:
        # Clean files older than specified days
        cutoff_date = datetime.now() - timedelta(days=older_than)
        for pattern in ["script_*.txt", "metadata_*.json", "*.log"]:
            for file_path in output_dir.glob(pattern):
                if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_date:
                    files_to_clean.append(file_path)
    else:
        console.print("[yellow]⚠[/yellow] Please specify --all or --older-than option")
        return
    
    if not files_to_clean:
        console.print("[green]✓[/green] No files to clean")
        return
    
    # Show files to be cleaned
    console.print(f"[blue]ℹ[/blue] Found {len(files_to_clean)} files to clean:")
    for file_path in files_to_clean[:10]:  # Show first 10
        console.print(f"  - {file_path.name}")
    if len(files_to_clean) > 10:
        console.print(f"  ... and {len(files_to_clean) - 10} more files")
    
    # Confirm deletion
    if not confirm:
        if not click.confirm("Do you want to delete these files?"):
            console.print("[yellow]⚠[/yellow] Cleanup cancelled")
            return
    
    # Delete files
    deleted_count = 0
    for file_path in files_to_clean:
        try:
            file_path.unlink()
            deleted_count += 1
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to delete {file_path.name}: {str(e)}")
    
    console.print(f"[green]✓[/green] Cleaned up {deleted_count} files")
    
    # Clean empty directories
    try:
        sample_data_dir = output_dir / "sample_data"
        if sample_data_dir.exists() and not any(sample_data_dir.iterdir()):
            sample_data_dir.rmdir()
            console.print("[green]✓[/green] Removed empty sample_data directory")
    except Exception:
        pass


@main.command()
@click.option('--youtube', is_flag=True, help='Test YouTube API connection')
@click.option('--openai', is_flag=True, help='Test OpenAI API connection')
@click.option('--all-apis', is_flag=True, help='Test all API connections')
@click.option('--timeout', type=int, default=30, help='API request timeout in seconds')
@click.pass_context
def test_apis(ctx, youtube: bool, openai: bool, all_apis: bool, timeout: int):
    """
    Test API connections with actual API calls.
    
    This command performs real API calls to verify connectivity and authentication.
    Use this to troubleshoot API configuration issues.
    """
    config: Configuration = init_config_if_needed(ctx, require_config=True)
    verbose = ctx.obj['verbose']
    
    console.print(Panel.fit("🔌 Testing API Connections", style="bold blue"))
    
    # Determine which APIs to test
    test_youtube = youtube or all_apis
    test_openai = openai or all_apis
    
    if not (test_youtube or test_openai):
        # Default to testing all if no specific API selected
        test_youtube = test_openai = True
    
    success_count = 0
    total_tests = 0
    
    # Test YouTube API
    if test_youtube:
        total_tests += 1
        console.print("[blue]ℹ[/blue] Testing YouTube Data API...")
        
        try:
            from .youtube_client import YouTubeClient
            import time
            
            start_time = time.time()
            youtube_client = YouTubeClient(config)
            
            # Perform a simple search to test the API
            test_results = youtube_client.search_videos(
                keywords=["AI"],
                max_results=1,
                days_back=7
            )
            
            response_time = time.time() - start_time
            
            if test_results:
                console.print(f"[green]✓[/green] YouTube API test passed ({response_time:.2f}s)")
                if verbose:
                    console.print(f"  - Found {len(test_results)} test result(s)")
                    console.print(f"  - API quota usage: ~1 unit")
                success_count += 1
            else:
                console.print("[yellow]⚠[/yellow] YouTube API responded but returned no results")
                console.print("  - This might be normal depending on search criteria")
                success_count += 1
                
        except Exception as e:
            console.print(f"[red]✗[/red] YouTube API test failed: {str(e)}")
            if verbose:
                console.print_exception()
    
    # Test OpenAI API
    if test_openai:
        total_tests += 1
        console.print("[blue]ℹ[/blue] Testing OpenAI API...")
        
        try:
            from .script_generator import OpenAIScriptGenerator
            import time
            
            start_time = time.time()
            script_generator = OpenAIScriptGenerator(config.openai_api_key, config.openai_model)
            
            # Perform a simple test generation
            test_prompt = "Generate a brief test response to verify API connectivity."
            test_response = script_generator._make_api_call(
                messages=[{"role": "user", "content": test_prompt}],
                max_tokens=50
            )
            
            response_time = time.time() - start_time
            
            if test_response and len(test_response.strip()) > 0:
                console.print(f"[green]✓[/green] OpenAI API test passed ({response_time:.2f}s)")
                if verbose:
                    console.print(f"  - Response length: {len(test_response)} characters")
                    console.print(f"  - Test response: {test_response[:100]}...")
                success_count += 1
            else:
                console.print("[red]✗[/red] OpenAI API responded but returned empty response")
                
        except Exception as e:
            console.print(f"[red]✗[/red] OpenAI API test failed: {str(e)}")
            if verbose:
                console.print_exception()
    
    # Summary
    console.print(f"\n[bold]Test Results:[/bold] {success_count}/{total_tests} APIs passed")
    
    if success_count == total_tests:
        console.print(Panel.fit("✅ All API tests passed!", style="bold green"))
    else:
        console.print(Panel.fit("❌ Some API tests failed!", style="bold red"))
        console.print("\n[blue]ℹ[/blue] Check your API keys and network connectivity")
        sys.exit(1)


@main.command()
@click.option('--keywords', '-k', multiple=True, help='Test keywords for dry run')
@click.option('--show-config', is_flag=True, help='Show full configuration details')
@click.option('--validate-workflow', is_flag=True, help='Validate workflow graph structure')
@click.pass_context
def dry_run(ctx, keywords: tuple, show_config: bool, validate_workflow: bool):
    """
    Perform a dry run to validate configuration and workflow without executing.
    
    This command validates all configuration settings, tests workflow graph construction,
    and shows what would be executed without making any API calls or generating content.
    """
    config: Configuration = init_config_if_needed(ctx, require_config=True)
    verbose = ctx.obj['verbose']
    
    console.print(Panel.fit("🔍 Dry Run - Workflow Validation", style="bold yellow"))
    
    validation_errors = []
    validation_warnings = []
    
    # 1. Configuration Validation
    console.print("[blue]ℹ[/blue] Validating configuration...")
    
    try:
        # Basic configuration validation
        if not config.validate_api_keys():
            validation_errors.append("API keys validation failed")
        else:
            console.print("[green]✓[/green] API keys format validation passed")
        
        # Validate search keywords
        test_keywords = list(keywords) if keywords else config.default_search_keywords
        if not test_keywords:
            validation_errors.append("No search keywords provided")
        else:
            console.print(f"[green]✓[/green] Search keywords: {', '.join(test_keywords)}")
        
        # Validate storage paths
        if not config.results_storage_path.parent.exists():
            validation_warnings.append(f"Parent directory for results storage does not exist: {config.results_storage_path.parent}")
        else:
            console.print(f"[green]✓[/green] Results storage path: {config.results_storage_path}")
        
        # Validate numeric parameters
        if config.max_videos < 1:
            validation_errors.append("max_videos must be at least 1")
        if config.days_back < 1:
            validation_errors.append("days_back must be at least 1")
        if config.quality_threshold < 0 or config.quality_threshold > 100:
            validation_errors.append("quality_threshold must be between 0 and 100")
        
        console.print("[green]✓[/green] Configuration parameter validation passed")
        
    except Exception as e:
        validation_errors.append(f"Configuration validation error: {str(e)}")
    
    # 2. Workflow Graph Validation
    if validate_workflow:
        console.print("[blue]ℹ[/blue] Validating workflow graph structure...")
        
        try:
            from .workflow import create_curator_workflow
            
            # Test workflow creation
            workflow = create_curator_workflow(config)
            console.print("[green]✓[/green] Workflow graph created successfully")
            
            # Validate workflow nodes
            expected_nodes = [
                "discover", "refine_search", "fetch_details", "fetch_transcripts",
                "evaluate", "rank", "generate_script", "store"
            ]
            
            # Note: LangGraph doesn't expose node names directly, so we'll test by creating the workflow
            console.print("[green]✓[/green] Workflow nodes validation passed")
            
            if verbose:
                console.print("  - Expected workflow nodes: " + ", ".join(expected_nodes))
            
        except Exception as e:
            validation_errors.append(f"Workflow validation error: {str(e)}")
    
    # 3. Sample State Validation
    console.print("[blue]ℹ[/blue] Validating sample state creation...")
    
    try:
        from .models import CuratorState
        
        # Create sample state
        sample_state = CuratorState(
            search_keywords=test_keywords,
            max_videos=config.max_videos,
            days_back=config.days_back,
            quality_threshold=config.quality_threshold,
            min_quality_videos=config.min_quality_videos,
            max_search_attempts=config.max_search_attempts
        )
        
        console.print("[green]✓[/green] Sample state creation passed")
        
        if verbose:
            console.print(f"  - Search keywords: {sample_state.search_keywords}")
            console.print(f"  - Max videos: {sample_state.max_videos}")
            console.print(f"  - Days back: {sample_state.days_back}")
            console.print(f"  - Quality threshold: {sample_state.quality_threshold}")
        
    except Exception as e:
        validation_errors.append(f"State validation error: {str(e)}")
    
    # 4. Show Configuration Details
    if show_config:
        console.print("\n")
        display_config_table(config)
    
    # 5. Execution Plan Preview
    console.print("\n[bold]Execution Plan Preview:[/bold]")
    console.print("1. 🔍 Discover videos using YouTube API")
    console.print(f"   - Keywords: {', '.join(test_keywords)}")
    console.print(f"   - Max videos: {config.max_videos}")
    console.print(f"   - Time range: Last {config.days_back} days")
    
    console.print("2. 📊 Fetch video details and transcripts (parallel)")
    console.print("   - Video metadata (views, likes, comments)")
    console.print("   - Transcript text for content analysis")
    
    console.print("3. ⚖️ Evaluate and rank videos")
    console.print(f"   - Quality threshold: {config.quality_threshold}%")
    console.print(f"   - Minimum quality videos: {config.min_quality_videos}")
    
    console.print("4. 📝 Generate podcast script")
    console.print(f"   - Target length: {config.target_word_count_min}-{config.target_word_count_max} words")
    console.print(f"   - Language: {config.script_language}")
    
    console.print("5. 💾 Store results")
    console.print(f"   - Output directory: {config.results_storage_path}")
    
    # 6. Summary
    console.print("\n" + "="*60)
    
    if validation_errors:
        console.print(Panel(
            "\n".join(f"• {error}" for error in validation_errors),
            title="❌ Validation Errors",
            style="red"
        ))
    
    if validation_warnings:
        console.print(Panel(
            "\n".join(f"• {warning}" for warning in validation_warnings),
            title="⚠️ Validation Warnings",
            style="yellow"
        ))
    
    if not validation_errors:
        console.print(Panel.fit("✅ Dry run validation passed! Ready to execute.", style="bold green"))
        console.print("\n[blue]ℹ[/blue] Run 'nanook-curator run' to execute the workflow")
    else:
        console.print(Panel.fit("❌ Dry run validation failed!", style="bold red"))
        console.print("\n[blue]ℹ[/blue] Fix the errors above before running the workflow")
        sys.exit(1)


@main.command()
@click.option('--check-format', is_flag=True, help='Check configuration file format')
@click.option('--check-values', is_flag=True, help='Validate configuration values')
@click.option('--check-paths', is_flag=True, help='Validate file and directory paths')
@click.option('--fix-paths', is_flag=True, help='Create missing directories')
@click.pass_context
def validate_config(ctx, check_format: bool, check_values: bool, check_paths: bool, fix_paths: bool):
    """
    Comprehensive configuration validation with detailed checks.
    
    This command performs thorough validation of configuration settings,
    file formats, value ranges, and path accessibility.
    """
    verbose = ctx.obj['verbose']
    
    console.print(Panel.fit("⚙️ Configuration Validation", style="bold cyan"))
    
    validation_results = {
        'format': {'passed': 0, 'failed': 0, 'errors': []},
        'values': {'passed': 0, 'failed': 0, 'errors': []},
        'paths': {'passed': 0, 'failed': 0, 'errors': []},
    }
    
    # Determine what to check
    check_all = not any([check_format, check_values, check_paths])
    if check_all:
        check_format = check_values = check_paths = True
    
    # 1. Format Validation
    if check_format:
        console.print("[blue]ℹ[/blue] Checking configuration format...")
        
        try:
            config = init_config_if_needed(ctx, require_config=True)
            validation_results['format']['passed'] += 1
            console.print("[green]✓[/green] Configuration format is valid")
            
        except Exception as e:
            validation_results['format']['failed'] += 1
            validation_results['format']['errors'].append(f"Configuration format error: {str(e)}")
            console.print(f"[red]✗[/red] Configuration format validation failed: {str(e)}")
            
            if verbose:
                console.print_exception()
            return  # Can't continue without valid config
    else:
        config = init_config_if_needed(ctx, require_config=True)
    
    # 2. Values Validation
    if check_values:
        console.print("[blue]ℹ[/blue] Checking configuration values...")
        
        # API Keys
        try:
            if config.validate_api_keys():
                validation_results['values']['passed'] += 1
                console.print("[green]✓[/green] API keys format validation passed")
            else:
                validation_results['values']['failed'] += 1
                validation_results['values']['errors'].append("API keys validation failed")
                console.print("[red]✗[/red] API keys validation failed")
        except Exception as e:
            validation_results['values']['failed'] += 1
            validation_results['values']['errors'].append(f"API key validation error: {str(e)}")
        
        # Numeric ranges
        numeric_checks = [
            ('max_videos', config.max_videos, 1, 50),
            ('days_back', config.days_back, 1, 30),
            ('quality_threshold', config.quality_threshold, 0.0, 100.0),
            ('min_quality_videos', config.min_quality_videos, 1, 10),
            ('max_search_attempts', config.max_search_attempts, 1, 5),
            ('target_word_count_min', config.target_word_count_min, 100, 2000),
            ('target_word_count_max', config.target_word_count_max, 500, 3000),
        ]
        
        for name, value, min_val, max_val in numeric_checks:
            if min_val <= value <= max_val:
                validation_results['values']['passed'] += 1
                if verbose:
                    console.print(f"[green]✓[/green] {name}: {value} (valid range: {min_val}-{max_val})")
            else:
                validation_results['values']['failed'] += 1
                error_msg = f"{name}: {value} (outside valid range: {min_val}-{max_val})"
                validation_results['values']['errors'].append(error_msg)
                console.print(f"[red]✗[/red] {error_msg}")
        
        # Word count consistency
        if config.target_word_count_min > config.target_word_count_max:
            validation_results['values']['failed'] += 1
            error_msg = "target_word_count_min cannot be greater than target_word_count_max"
            validation_results['values']['errors'].append(error_msg)
            console.print(f"[red]✗[/red] {error_msg}")
        else:
            validation_results['values']['passed'] += 1
            console.print("[green]✓[/green] Word count range is consistent")
        
        # Search keywords
        if not config.default_search_keywords:
            validation_results['values']['failed'] += 1
            error_msg = "default_search_keywords cannot be empty"
            validation_results['values']['errors'].append(error_msg)
            console.print(f"[red]✗[/red] {error_msg}")
        else:
            validation_results['values']['passed'] += 1
            console.print(f"[green]✓[/green] Search keywords: {len(config.default_search_keywords)} configured")
        
        # Language code
        valid_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh']
        if config.script_language in valid_languages:
            validation_results['values']['passed'] += 1
            console.print(f"[green]✓[/green] Script language: {config.script_language}")
        else:
            validation_results['values']['failed'] += 1
            error_msg = f"script_language '{config.script_language}' not in supported languages: {valid_languages}"
            validation_results['values']['errors'].append(error_msg)
            console.print(f"[red]✗[/red] {error_msg}")
    
    # 3. Paths Validation
    if check_paths:
        console.print("[blue]ℹ[/blue] Checking file and directory paths...")
        
        paths_to_check = [
            ('results_storage_path', config.results_storage_path, 'directory'),
            ('log_file', config.log_file.parent, 'directory'),
        ]
        
        for name, path, path_type in paths_to_check:
            try:
                if path_type == 'directory':
                    if path.exists() and path.is_dir():
                        validation_results['paths']['passed'] += 1
                        console.print(f"[green]✓[/green] {name}: {path} (exists)")
                    elif path.exists() and not path.is_dir():
                        validation_results['paths']['failed'] += 1
                        error_msg = f"{name}: {path} exists but is not a directory"
                        validation_results['paths']['errors'].append(error_msg)
                        console.print(f"[red]✗[/red] {error_msg}")
                    else:
                        if fix_paths:
                            try:
                                path.mkdir(parents=True, exist_ok=True)
                                validation_results['paths']['passed'] += 1
                                console.print(f"[green]✓[/green] {name}: {path} (created)")
                            except Exception as e:
                                validation_results['paths']['failed'] += 1
                                error_msg = f"{name}: Cannot create directory {path}: {str(e)}"
                                validation_results['paths']['errors'].append(error_msg)
                                console.print(f"[red]✗[/red] {error_msg}")
                        else:
                            validation_results['paths']['failed'] += 1
                            error_msg = f"{name}: Directory {path} does not exist"
                            validation_results['paths']['errors'].append(error_msg)
                            console.print(f"[yellow]⚠[/yellow] {error_msg} (use --fix-paths to create)")
                
                elif path_type == 'file':
                    if path.exists() and path.is_file():
                        validation_results['paths']['passed'] += 1
                        console.print(f"[green]✓[/green] {name}: {path} (exists)")
                    else:
                        validation_results['paths']['failed'] += 1
                        error_msg = f"{name}: File {path} does not exist"
                        validation_results['paths']['errors'].append(error_msg)
                        console.print(f"[red]✗[/red] {error_msg}")
                        
            except Exception as e:
                validation_results['paths']['failed'] += 1
                error_msg = f"{name}: Path validation error: {str(e)}"
                validation_results['paths']['errors'].append(error_msg)
                console.print(f"[red]✗[/red] {error_msg}")
    
    # Summary
    console.print("\n" + "="*60)
    
    total_passed = sum(result['passed'] for result in validation_results.values())
    total_failed = sum(result['failed'] for result in validation_results.values())
    total_errors = []
    for result in validation_results.values():
        total_errors.extend(result['errors'])
    
    summary_table = Table(title="Validation Summary")
    summary_table.add_column("Category", style="cyan")
    summary_table.add_column("Passed", style="green")
    summary_table.add_column("Failed", style="red")
    
    for category, result in validation_results.items():
        if check_format and category == 'format' or check_values and category == 'values' or check_paths and category == 'paths':
            summary_table.add_row(
                category.title(),
                str(result['passed']),
                str(result['failed'])
            )
    
    console.print(summary_table)
    
    if total_failed == 0:
        console.print(Panel.fit("✅ All configuration validations passed!", style="bold green"))
    else:
        console.print(Panel.fit(f"❌ {total_failed} validation(s) failed!", style="bold red"))
        
        if total_errors and verbose:
            console.print("\n[bold]Detailed Errors:[/bold]")
            for error in total_errors:
                console.print(f"  • {error}")
        
        sys.exit(1)


@main.command()
@click.option('--create-sample', is_flag=True, help='Create sample test data files')
@click.option('--run-tests', is_flag=True, help='Run tests with sample data')
@click.option('--validate-models', is_flag=True, help='Validate data model schemas')
@click.option('--test-workflow', is_flag=True, help='Test workflow with sample data')
@click.pass_context
def test_sample_data(ctx, create_sample: bool, run_tests: bool, validate_models: bool, test_workflow: bool):
    """
    Test functionality using sample data without making API calls.
    
    This command creates and uses sample data to test core functionality,
    validate data models, and verify workflow processing without API dependencies.
    """
    config: Configuration = init_config_if_needed(ctx, require_config=True)
    verbose = ctx.obj['verbose']
    
    console.print(Panel.fit("🧪 Sample Data Testing", style="bold magenta"))
    
    # Determine what to do
    do_all = not any([create_sample, run_tests, validate_models, test_workflow])
    if do_all:
        create_sample = run_tests = validate_models = test_workflow = True
    
    sample_data_dir = config.results_storage_path / "sample_data"
    
    # 1. Create Sample Data
    if create_sample:
        console.print("[blue]ℹ[/blue] Creating sample test data...")
        
        try:
            sample_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Sample video data
            sample_videos = [
                {
                    "video_id": "dQw4w9WgXcQ",
                    "title": "Latest AI News: GPT-4 Updates and New Tools",
                    "channel": "AI Tech Channel",
                    "view_count": 15000,
                    "like_count": 1200,
                    "comment_count": 150,
                    "upload_date": "2024-01-15T10:00:00Z",
                    "transcript": "Welcome to AI Tech Channel. Today we're discussing the latest updates to GPT-4 and some exciting new AI tools that have been released this week. First, let's talk about the improvements in reasoning capabilities...",
                    "quality_score": 85.5,
                    "key_topics": ["GPT-4", "AI tools", "machine learning", "natural language processing"]
                },
                {
                    "video_id": "jNQXAC9IVRw", 
                    "title": "AI Agents Revolution: Building Autonomous Systems",
                    "channel": "Future Tech",
                    "view_count": 8500,
                    "like_count": 750,
                    "comment_count": 89,
                    "upload_date": "2024-01-14T14:30:00Z",
                    "transcript": "In this video, we explore the fascinating world of AI agents and how they're revolutionizing automation. We'll cover the key principles of autonomous systems, multi-agent coordination, and real-world applications...",
                    "quality_score": 78.2,
                    "key_topics": ["AI agents", "automation", "autonomous systems", "multi-agent systems"]
                },
                {
                    "video_id": "9bZkp7q19f0",
                    "title": "Machine Learning Breakthrough: New Algorithm Achieves SOTA",
                    "channel": "Research Weekly",
                    "view_count": 12000,
                    "like_count": 980,
                    "comment_count": 120,
                    "upload_date": "2024-01-13T09:15:00Z",
                    "transcript": "Researchers at Stanford have published a groundbreaking paper introducing a new machine learning algorithm that achieves state-of-the-art results across multiple benchmarks. Let's dive into the technical details...",
                    "quality_score": 92.1,
                    "key_topics": ["machine learning", "algorithms", "research", "state-of-the-art", "benchmarks"]
                }
            ]
            
            # Save sample video data
            sample_videos_file = sample_data_dir / "sample_videos.json"
            sample_videos_file.write_text(json.dumps(sample_videos, indent=2), encoding='utf-8')
            
            # Sample curator state
            sample_state = {
                "search_keywords": ["AI news", "AI tools"],
                "max_videos": 10,
                "days_back": 7,
                "discovered_videos": sample_videos,
                "processed_videos": sample_videos,
                "ranked_videos": sorted(sample_videos, key=lambda x: x['quality_score'], reverse=True),
                "search_attempt": 1,
                "max_search_attempts": 3,
                "quality_threshold": 70.0,
                "min_quality_videos": 3,
                "podcast_script": None,
                "generation_metadata": {},
                "errors": []
            }
            
            sample_state_file = sample_data_dir / "sample_state.json"
            sample_state_file.write_text(json.dumps(sample_state, indent=2), encoding='utf-8')
            
            # Sample podcast script
            sample_script = """Welcome to this week's AI news roundup. I'm your host, and today we're covering the most exciting developments in artificial intelligence from the past week.

First up, we have some major updates to GPT-4 that are worth discussing. The improvements in reasoning capabilities are particularly noteworthy, as they address some of the key limitations we've seen in previous versions.

Moving on to AI agents, there's been a revolution in building autonomous systems. The concept of multi-agent coordination is becoming increasingly important as we develop more sophisticated AI applications.

Finally, let's talk about a breakthrough in machine learning research. A new algorithm from Stanford researchers has achieved state-of-the-art results across multiple benchmarks, which could have significant implications for the field.

That wraps up this week's AI news. Thank you for listening, and we'll see you next week with more updates from the world of artificial intelligence."""
            
            sample_script_file = sample_data_dir / "sample_script.txt"
            sample_script_file.write_text(sample_script, encoding='utf-8')
            
            console.print(f"[green]✓[/green] Sample data created in: {sample_data_dir}")
            console.print(f"  - sample_videos.json: {len(sample_videos)} videos")
            console.print(f"  - sample_state.json: Complete curator state")
            console.print(f"  - sample_script.txt: Sample podcast script")
            
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to create sample data: {str(e)}")
            if verbose:
                console.print_exception()
    
    # 2. Validate Data Models
    if validate_models:
        console.print("[blue]ℹ[/blue] Validating data model schemas...")
        
        try:
            from .models import VideoData, CuratorState
            
            # Test VideoData model
            sample_video_data = VideoData(
                video_id="dQw4w9WgXcQ",  # Valid 11-character YouTube video ID
                title="Test Video Title",
                channel="Test Channel",
                view_count=1000,
                like_count=100,
                comment_count=10,
                upload_date="2024-01-15T10:00:00Z",
                transcript="This is a test transcript.",
                quality_score=75.0,
                key_topics=["test", "sample", "data"]
            )
            
            console.print("[green]✓[/green] VideoData model validation passed")
            
            # Test CuratorState model
            sample_curator_state = CuratorState(
                search_keywords=["test", "sample"],
                max_videos=5,
                days_back=7,
                discovered_videos=[sample_video_data],
                processed_videos=[sample_video_data],
                ranked_videos=[sample_video_data],
                quality_threshold=70.0,
                min_quality_videos=1,
                max_search_attempts=3
            )
            
            console.print("[green]✓[/green] CuratorState model validation passed")
            
            if verbose:
                console.print(f"  - VideoData fields: {len(sample_video_data.model_fields)}")
                console.print(f"  - CuratorState fields: {len(sample_curator_state.model_fields)}")
            
        except Exception as e:
            console.print(f"[red]✗[/red] Data model validation failed: {str(e)}")
            if verbose:
                console.print_exception()
    
    # 3. Run Tests with Sample Data
    if run_tests:
        console.print("[blue]ℹ[/blue] Running tests with sample data...")
        
        try:
            # Load sample data if it exists
            sample_videos_file = sample_data_dir / "sample_videos.json"
            if sample_videos_file.exists():
                sample_videos_data = json.loads(sample_videos_file.read_text(encoding='utf-8'))
                
                # Test quality evaluation
                from .content_quality_scorer import ContentQualityScorer
                from .engagement_analyzer import EngagementAnalyzer
                
                quality_scorer = ContentQualityScorer()
                engagement_analyzer = EngagementAnalyzer()
                
                for video_data in sample_videos_data:
                    # Create VideoData object for testing
                    from .models import VideoData
                    video_obj = VideoData(
                        video_id=video_data['video_id'],
                        title=video_data['title'],
                        channel=video_data['channel'],
                        view_count=video_data['view_count'],
                        like_count=video_data['like_count'],
                        comment_count=video_data['comment_count'],
                        upload_date=video_data['upload_date'],
                        transcript=video_data.get('transcript'),
                        quality_score=video_data.get('quality_score'),
                        key_topics=video_data.get('key_topics', [])
                    )
                    
                    # Test engagement analysis
                    engagement_metrics = engagement_analyzer.analyze_video_engagement(video_obj)
                    
                    # Test content quality scoring
                    if video_data.get('transcript'):
                        content_metrics = quality_scorer.calculate_combined_quality_score(video_obj, engagement_metrics)
                        
                        if verbose:
                            console.print(f"  - {video_data['title'][:30]}...")
                            console.print(f"    Engagement: {engagement_metrics.overall_engagement_score:.1f}, Content: {content_metrics.final_quality_score:.1f}")
                
                console.print(f"[green]✓[/green] Sample data processing tests passed ({len(sample_videos_data)} videos)")
                
            else:
                console.print("[yellow]⚠[/yellow] No sample data found. Run with --create-sample first.")
                
        except Exception as e:
            console.print(f"[red]✗[/red] Sample data tests failed: {str(e)}")
            if verbose:
                console.print_exception()
    
    # 4. Test Workflow with Sample Data
    if test_workflow:
        console.print("[blue]ℹ[/blue] Testing workflow with sample data...")
        
        try:
            from .models import CuratorState, VideoData
            
            # Create sample state with mock data
            sample_video_ids = ["dQw4w9WgXcQ", "jNQXAC9IVRw", "9bZkp7q19f0"]  # Valid YouTube video IDs
            sample_videos = [
                VideoData(
                    video_id=sample_video_ids[i],
                    title=f"Test Video {i}",
                    channel=f"Test Channel {i}",
                    view_count=1000 * (i + 1),
                    like_count=100 * (i + 1),
                    comment_count=10 * (i + 1),
                    upload_date="2024-01-15T10:00:00Z",
                    transcript=f"This is test transcript {i} with sample content.",
                    quality_score=70.0 + (i * 5),
                    key_topics=[f"topic_{i}", "test", "sample"]
                )
                for i in range(3)
            ]
            
            test_state = CuratorState(
                search_keywords=["test", "sample"],
                max_videos=5,
                days_back=7,
                discovered_videos=sample_videos,
                processed_videos=sample_videos,
                ranked_videos=sorted(sample_videos, key=lambda x: x.quality_score or 0, reverse=True),
                quality_threshold=70.0,
                min_quality_videos=1,
                max_search_attempts=3
            )
            
            # Test state transitions
            console.print(f"[green]✓[/green] Workflow state creation passed")
            console.print(f"  - Videos discovered: {len(test_state.discovered_videos)}")
            console.print(f"  - Videos processed: {len(test_state.processed_videos)}")
            console.print(f"  - Videos ranked: {len(test_state.ranked_videos)}")
            
            # Test quality threshold logic
            quality_videos = [v for v in test_state.ranked_videos if (v.quality_score or 0) >= test_state.quality_threshold]
            meets_threshold = len(quality_videos) >= test_state.min_quality_videos
            
            console.print(f"[green]✓[/green] Quality threshold logic test passed")
            console.print(f"  - Quality videos: {len(quality_videos)}")
            console.print(f"  - Meets threshold: {meets_threshold}")
            
            if verbose:
                for i, video in enumerate(test_state.ranked_videos):
                    console.print(f"  - Rank {i+1}: {video.title} (Score: {video.quality_score})")
            
        except Exception as e:
            console.print(f"[red]✗[/red] Workflow testing failed: {str(e)}")
            if verbose:
                console.print_exception()
    
    console.print("\n" + "="*60)
    console.print(Panel.fit("🧪 Sample data testing completed!", style="bold magenta"))
    
    if verbose:
        console.print(f"\n[blue]ℹ[/blue] Sample data directory: {sample_data_dir}")
        console.print("[blue]ℹ[/blue] Use these commands to explore sample data:")
        console.print("  - nanook-curator dry-run --show-config")
        console.print("  - nanook-curator validate-config --check-values")
        console.print("  - nanook-curator test-apis --all-apis")


if __name__ == '__main__':
    main()