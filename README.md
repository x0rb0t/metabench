# MetaBench - Self-Evaluating Transformation Benchmark

A sophisticated benchmarking tool that evaluates LLM performance on content transformation tasks with enhanced instruction diversity, creative content generation, proper scoring logic, and comprehensive logging.

## What is MetaBench?

MetaBench challenges language models with increasingly complex transformation tasks - from simple format conversions to multi-layered data restructuring with conditional logic. It generates creative content, applies sophisticated transformation instructions, and provides detailed scoring on how well the LLM follows complex, multi-step requirements.

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
from main import BenchmarkConfig, TransformationBenchmark

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
```bash
export METABENCH_API_KEY="your-api-key"
export METABENCH_BASE_URL="https://api.openai.com/v1"
export METABENCH_MODEL="gpt-4"
```

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
uv run main.py [OPTIONS]

Options:
  -i, --interactive          Run in interactive mode
  -q, --quick               Quick benchmark (2 levels, 1 trial each)
  --url TEXT                Base URL for LLM API
  --api-key TEXT            API key for LLM service
  --model TEXT              Model name
  --complexity INTEGER      Maximum complexity level (1-5)
  --trials INTEGER          Number of trials per complexity
  --content TEXT            Content types: code,text,data,configuration,documentation
  --topic TEXT              Topic for content generation
  --temperature FLOAT       Temperature for LLM (0.0-2.0)
  --output TEXT             Output filename for results
  --log-file TEXT           Log filename
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
├── main.py              # Complete benchmark implementation
├── pyproject.toml       # uv project configuration
├── README.md           # This documentation
├── .gitignore          # Git ignore rules
├── uv.lock            # Dependency lock file
└── logs/              # Auto-created benchmark logs
    ├── benchmark_log_*.log
└── benchmark_results/  # Auto-created benchmark results
    └── benchmark_results_*.json
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Test with `uv run main.py --quick`
4. Submit a pull request

## License

MIT License - see LICENSE file for details.