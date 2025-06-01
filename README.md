# MetaBench - Self-Evaluating Transformation Benchmark

A sophisticated benchmarking tool that evaluates LLM performance on content transformation tasks with enhanced instruction diversity, creative content generation, proper scoring logic, and comprehensive logging.

## What is MetaBench?

MetaBench challenges language models with increasingly complex transformation tasks - from simple format conversions to multi-layered data restructuring with conditional logic. It generates creative content, applies sophisticated transformation instructions, and provides detailed scoring on how well the LLM follows complex, multi-step requirements.

## How It Works

MetaBench follows a simple 4-step process for each trial:

1. **Generate Content** → 2. **Create Instructions** → 3. **Transform** → 4. **Score**

### Toy Example

**Step 1: Generate Content**
```
Original content: "Alice: 25, Bob: 30, Carol: 28"
```

**Step 2: Create Transformation Instructions**
```
Convert to JSON format with ages grouped by decade
```

**Step 3: LLM Transforms**
```json
{
  "twenties": [
    {"name": "Alice", "age": 25},
    {"name": "Carol", "age": 28}
  ],
  "thirties": [
    {"name": "Bob", "age": 30}
  ]
}
```

**Step 4: Score the Result**
- ✅ Followed JSON format: 10/10
- ✅ Grouped by decade correctly: 10/10  
- ✅ Preserved all data: 10/10
- **Final Score: 10/10**

As complexity increases, instructions become multi-layered with conditional logic, cross-references, and validation requirements.

## Architecture

MetaBench is built with a modular architecture for maintainability and extensibility:

### Core Modules

- **`main.py`** - Entry point with CLI handling and main execution flow
- **`modules/benchmark.py`** - Main benchmark orchestration and execution logic
- **`modules/config.py`** - Configuration data classes and validation
- **`modules/cli.py`** - Command-line argument parsing and interactive mode
- **`modules/llm.py`** - LLM wrapper with retry logic and error handling
- **`modules/generators.py`** - Content and instruction generation engines
- **`modules/engines.py`** - Transformation and verification engines
- **`modules/parsers.py`** - Custom output parsers for structured responses
- **`modules/utils.py`** - Utility functions (env resolution, logging setup)

This modular design allows for easy testing, modification, and extension of individual components without affecting the entire system.

## Features

- **🎯 Enhanced Instruction Generation**: Creates diverse transformation instructions with varying complexity levels (1-5)
- **🎨 Creative Content Generation**: Generates varied content with random enhancements (unicode symbols, emojis, special formatting)
- **⚙️ Comprehensive Transformation Engine**: Applies complex transformation instructions to content
- **🔍 Advanced Verification**: Scores transformation results with detailed criteria and specific scoring categories
- **📝 Dual Logging**: File-based detailed logging and console emoji output for easy monitoring
- **📊 Multiple Content Types**: Supports code, text, data, configuration, and documentation
- **📈 Progress Tracking**: Real-time progress updates with trial completion status
- **🛡️ Robust Error Handling**: Network, JSON, and unexpected error handling with graceful degradation

## Quick Start

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and run
git clone <your-repo>
cd metabench
uv run main.py --quick
```

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. If you don't have uv installed, choose one of the methods below:

### Installing uv

#### Standalone Installer (Recommended)

**macOS and Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Alternative Installation Methods

**Homebrew (macOS):**
```bash
brew install uv
```

**PyPI (with pipx):**
```bash
pipx install uv
```

**PyPI (with pip):**
```bash
pip install uv
```

**WinGet (Windows):**
```bash
winget install --id=astral-sh.uv -e
```

**Scoop (Windows):**
```bash
scoop install main/uv
```

**Cargo:**
```bash
cargo install --git https://github.com/astral-sh/uv uv
```

### Setting up the Project

Once uv is installed:

```bash
# Install dependencies
uv sync

# Or run directly (uv will handle dependencies automatically)
uv run main.py
```

## Usage Examples

### Basic Usage
```bash
# Quick test (2 complexity levels, 1 trial each)
uv run main.py --quick

# Full benchmark with default settings
uv run main.py

# Custom complexity and trials
uv run main.py --complexity 4 --trials 5
```

### Model Configuration
```bash
# Local model (default)
uv run main.py --url http://localhost:1234/v1

# OpenAI API
uv run main.py --url https://api.openai.com/v1 --api-key sk-your-key --model gpt-4

# Anthropic Claude
uv run main.py --url https://api.anthropic.com --api-key your-key --model claude-3-sonnet
```

### Content-Specific Testing
```bash
# Test only code transformations
uv run main.py --content code --topic "machine learning"

# Focus on specific content types
uv run main.py --content "code,documentation" --complexity 3

# Topic-focused benchmark
uv run main.py --topic "blockchain" --trials 2
```

### Interactive Mode
```bash
# Guided setup with prompts
uv run main.py --interactive
```

## Sample Output

### Console Output
```
🚀 Starting Transformation Benchmark
Configuration: 3 complexity levels, 2 trials each
Content types: code, text, data
Total trials: 6
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Complexity Level 1

  Trial 1/6 (16.7%) - code
    🚀 Starting trial: Complexity 1, Type code
    🎯 Target format: nested JSON with metadata and timestamps
    🎨 Creative variations: add emoji and modern text symbols 🚀, include technical jargon
    ✅ Generated 342 chars with 2 creative variations
    ✅ Verification complete: Quality=8.50/10, Completion=92.0%
    ⏱️  Trial completed in 12.3s

📈 BENCHMARK SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Trials: 6
Average Quality Score: 7.83/10
Average Completion Rate: 89.2%
Error Rate: 0.0%
Total Time: 67.2s
```

### Detailed Scoring Example
```
    ✅ Verification complete: Quality=8.50/10, Completion=92.0%
      🎯 basic_rules_score: 9.20/10
      🎯 conditional_operations_score: 8.10/10
      ⚫ advanced_processing_score: N/A (not applicable)
      🎯 verification_requirements_score: 8.70/10
      🎯 data_preservation_score: 9.50/10
      🎯 format_compliance_score: 8.90/10
    📈 Average applicable score: 8.88/10
```

## Configuration

### Programmatic Configuration
```python
from modules import BenchmarkConfig, TransformationBenchmark

config = BenchmarkConfig(
    base_url="http://localhost:1234/v1",
    api_key="your-api-key",
    max_complexity=4,
    trials_per_complexity=3,
    content_types=["code", "text", "data"],
    topic="artificial intelligence",
    temperature=0.3
)

benchmark = TransformationBenchmark(config)
summary = benchmark.run_benchmark()
```

### Environment Variables

You can reference environment variables using the `env:` prefix in command-line arguments:

```bash
# Set environment variables
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_API_KEY="sk-your-openai-key"

# Reference them with env: prefix
uv run main.py --api-key env:OPENROUTER_API_KEY --url env:OPENROUTER_BASE_URL --model "deepseek/deepseek-r1"
uv run main.py --api-key env:OPENAI_API_KEY --url "https://api.openai.com/v1" --model "gpt-4"
```

You can also use a `.env` file (automatically loaded):
```bash
# .env file
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-your-openai-key
```

**Supported environment variable formats:**
- `--api-key env:OPENROUTER_API_KEY` - Uses `$OPENROUTER_API_KEY`
- `--url env:OPENROUTER_BASE_URL` - Uses `$OPENROUTER_BASE_URL`
- `--api-key env:OPENAI_API_KEY` - Uses `$OPENAI_API_KEY`

## Understanding Complexity Levels

| Level | Description | Features |
|-------|-------------|----------|
| 1 | Basic formatting | Simple rules, basic transformations |
| 2 | Conditional logic | If/then operations, basic metadata |
| 3 | Advanced processing | Multiple conditions, generated IDs |
| 4 | Cross-references | Bidirectional links, validation |
| 5 | Full complexity | Security, optimization, multiple outputs |

## CLI Reference

```bash
% uv run main.py --help
usage: main.py [-h] [--interactive] [--quick] [--url URL] [--api-key API_KEY] [--model MODEL] [--creative-model CREATIVE_MODEL]
               [--structured-model STRUCTURED_MODEL] [--transform-model TRANSFORM_MODEL] [--complexity COMPLEXITY] [--trials TRIALS]
               [--temperature TEMPERATURE] [--content CONTENT] [--topic TOPIC] [--output OUTPUT] [--log-file LOG_FILE] [--quiet]

Enhanced Self-Evaluating Transformation Benchmark

options:
  -h, --help            show this help message and exit
  --interactive, -i     Run in interactive mode
  --quick, -q           Quick benchmark (2 complexity levels, 1 trial each)
  --url URL, --base-url URL
                        Base URL for the LLM API
  --api-key API_KEY     API key for the LLM service
  --model MODEL, --model-name MODEL
                        Model name (leave empty for local models)
  --creative-model CREATIVE_MODEL
                        Specific model for creative content generation
  --structured-model STRUCTURED_MODEL
                        Specific model for structured tasks
  --transform-model TRANSFORM_MODEL
                        Specific model for transformations
  --complexity COMPLEXITY, --max-complexity COMPLEXITY
                        Maximum complexity level (1-5)
  --trials TRIALS, --trials-per-complexity TRIALS
                        Number of trials per complexity level
  --temperature TEMPERATURE
                        Base temperature for LLM
  --content CONTENT, --content-types CONTENT
                        Comma-separated list of content types
  --topic TOPIC         Topic for content generation (makes content topic-specific)
  --output OUTPUT, -o OUTPUT
                        Output filename for results (auto-generated if not specified)
  --log-file LOG_FILE   Log filename (auto-generated if not specified)
  --quiet               Reduce output verbosity

Examples:
  # Quick test with local model
  python benchmark.py --quick
  
  # Full benchmark with specific model
  python benchmark.py --model gpt-3.5-turbo --url https://api.openai.com/v1
  
  # Topic-focused benchmark with logging
  python benchmark.py --topic "blockchain" --content code,text --log-file blockchain_test.log
  
  # Interactive mode
  python benchmark.py --interactive
```

## Output Files

### Log Files (`logs/`)
- **Detailed logs**: `benchmark_log_YYYYMMDD_HHMMSS.log`

### Results (`results/`)
- **JSON results**: `benchmark_results_YYYYMMDD_HHMMSS.json`
```json
{
  "config": { /* sanitized configuration */ },
  "results": [ /* detailed trial results */ ],
  "summary": {
    "total_trials": 15,
    "average_score": 7.83,
    "average_completion_rate": 0.892,
    "scores_by_complexity": { "1": 8.2, "2": 7.8, "3": 7.4 }
  }
}
```

## Troubleshooting

### Common Issues

**Connection errors:**
```bash
# Check if your LLM server is running
curl http://localhost:1234/v1/models

# Test with a simple request
uv run main.py --quick --trials 1
```

**JSON parsing errors:**
- Usually indicates the LLM is not following JSON format requirements
- Try reducing complexity or adjusting temperature
- Check logs for full LLM responses

**Low scores:**
- Normal for higher complexity levels
- Ensure your model has sufficient context length
- Consider using more capable models for complex transformations

### Debug Mode
```bash
# Run with detailed logging
uv run main.py --quick --log-file debug.log

# Check the logs directory
ls -la logs/
```

## Requirements

- **Python**: 3.12+
- **Dependencies**: LangChain Core, OpenAI packages (handled by uv)
- **LLM API**: Compatible endpoint (local or remote)
- **Memory**: ~500MB for typical runs
- **Storage**: ~50MB per benchmark run (logs + results)

## Project Structure

```
metabench/
├── main.py                   # Main entry point and CLI interface
├── pyproject.toml           # uv project configuration
├── README.md               # This documentation
├── .gitignore              # Git ignore rules
├── uv.lock                # Dependency lock file
├── modules/               # Modular components
│   ├── __init__.py        # Module exports and package definition
│   ├── benchmark.py       # Main benchmark runner class
│   ├── cli.py            # CLI argument parsing and interactive mode
│   ├── config.py         # Configuration classes and validation
│   ├── engines.py        # Transformation and verification engines
│   ├── generators.py     # Content and instruction generators
│   ├── llm.py           # LLM wrapper functionality
│   ├── parsers.py       # Custom output parsers
│   └── utils.py         # Environment variable resolution and utilities
├── logs/                 # Auto-created benchmark logs
│   └── benchmark_log_*.log
└── results/             # Auto-created benchmark results
    └── benchmark_results_*.json
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Test with `uv run main.py --quick`
4. Submit a pull request

## License

MIT License - see LICENSE file for details.