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
    
    # Summary statistics
    stats_table = Table(title="Execution Summary")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="magenta")
    
    stats_table.add_row("Execution Time", f"{execution_time:.2f} seconds")
    stats_table.add_row("Videos Discovered", str(len(state.discovered_videos)))
    stats_table.add_row("Videos Processed", str(len(state.processed_videos)))
    stats_table.add_row("Videos Ranked", str(len(state.ranked_videos)))
    stats_table.add_row("Search Attempts", str(state.search_attempt))
    stats_table.add_row("Errors Encountered", str(len(state.errors)))
    
    if state.podcast_script:
        word_count = len(state.podcast_script.split())
        stats_table.add_row("Script Word Count", str(word_count))
        estimated_duration = word_count / 150  # ~150 words per minute
        stats_table.add_row("Estimated Duration", f"{estimated_duration:.1f} minutes")
    
    console.print(stats_table)
    
    # Show top ranked videos if available
    if state.ranked_videos:
        console.print("\n")
        videos_table = Table(title="Top Ranked Videos")
        videos_table.add_column("Rank", style="cyan")
        videos_table.add_column("Title", style="white", max_width=40)
        videos_table.add_column("Channel", style="yellow")
        videos_table.add_column("Quality Score", style="green")
        videos_table.add_column("Views", style="blue")
        
        for i, video in enumerate(state.ranked_videos[:5], 1):
            videos_table.add_row(
                str(i),
                video.title[:37] + "..." if len(video.title) > 40 else video.title,
                video.channel,
                f"{video.quality_score:.1f}" if video.quality_score else "N/A",
                f"{video.view_count:,}"
            )
        
        console.print(videos_table)
    
    # Show errors if any
    if state.errors:
        console.print("\n")
        console.print(Panel(
            "\n".join(state.errors),
            title="⚠️  Errors Encountered",
            style="yellow"
        ))


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
@click.option('--output-file', '-o', type=click.Path(path_type=Path),
              help='Output file for generated script')
@click.option('--dry-run', is_flag=True,
              help='Validate configuration and show what would be executed without running')
@click.pass_context
def run(ctx, keywords: tuple, max_videos: Optional[int], days_back: Optional[int],
        quality_threshold: Optional[float], output_file: Optional[Path], dry_run: bool):
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
        
        # Display results
        display_execution_results(final_state, execution_time)
        
        # Save script to file if specified
        if output_file and final_state.podcast_script:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(final_state.podcast_script, encoding='utf-8')
            console.print(f"[green]✓[/green] Script saved to: {output_file}")
        elif final_state.podcast_script:
            # Save to default location
            default_output = config.results_storage_path / f"script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            default_output.write_text(final_state.podcast_script, encoding='utf-8')
            console.print(f"[green]✓[/green] Script saved to: {default_output}")
        
        # Save metadata
        metadata_file = config.results_storage_path / f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        metadata = {
            'execution_time': execution_time,
            'search_keywords': search_keywords,
            'videos_discovered': len(final_state.discovered_videos),
            'videos_processed': len(final_state.processed_videos),
            'videos_ranked': len(final_state.ranked_videos),
            'search_attempts': final_state.search_attempt,
            'errors': final_state.errors,
            'generation_metadata': final_state.generation_metadata
        }
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        
        if verbose:
            console.print(f"[blue]ℹ[/blue] Metadata saved to: {metadata_file}")
        
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
        script_generator = OpenAIScriptGenerator(config.openai_api_key)
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
    
    # Find files to clean
    files_to_clean = []
    
    if clean_all:
        files_to_clean.extend(output_dir.glob("script_*.txt"))
        files_to_clean.extend(output_dir.glob("metadata_*.json"))
        files_to_clean.extend(output_dir.glob("*.log"))
    elif older_than:
        cutoff_date = datetime.now() - timedelta(days=older_than)
        
        for pattern in ["script_*.txt", "metadata_*.json", "*.log"]:
            for file_path in output_dir.glob(pattern):
                try:
                    file_date = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_date < cutoff_date:
                        files_to_clean.append(file_path)
                except Exception:
                    continue
    else:
        console.print("[red]✗[/red] Please specify --all or --older-than option")
        return
    
    if not files_to_clean:
        console.print("[green]✓[/green] No files to clean")
        return
    
    # Show files to be cleaned
    console.print(f"[yellow]⚠[/yellow] Found {len(files_to_clean)} files to clean:")
    for file_path in files_to_clean:
        console.print(f"  - {file_path.name}")
    
    # Confirmation
    if not confirm:
        if not click.confirm("Are you sure you want to delete these files?"):
            console.print("Cleanup cancelled")
            return
    
    # Clean files
    cleaned_count = 0
    for file_path in files_to_clean:
        try:
            file_path.unlink()
            cleaned_count += 1
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to delete {file_path.name}: {e}")
    
    console.print(f"[green]✓[/green] Cleaned {cleaned_count} files")


if __name__ == '__main__':
    main()