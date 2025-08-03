# nanook-curator

AI-powered content curation system that automatically discovers trending YouTube videos about AI news, tools, and agents, analyzes their content through transcripts, and generates curated podcast scripts of 5-10 minutes in length.

## Features

- Automated YouTube video discovery with trending analysis
- Transcript fetching and content analysis
- Quality evaluation based on engagement metrics
- Podcast script generation using AI
- Configurable scheduling and automation
- LangGraph-based workflow orchestration

## Installation

```bash
# Install uv if you haven't already
pip install uv

# Clone the repository
git clone <repository-url>
cd nanook-curator

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

## Configuration

Copy `.env.example` to `.env` and configure your API keys:

```bash
cp .env.example .env
```

## Usage

```bash
# Run the curator
nanook-curator

# Run with custom configuration
nanook-curator --config config.yaml
```

## Development

```bash
# Install development dependencies
uv sync --extra dev

# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

## License

MIT License