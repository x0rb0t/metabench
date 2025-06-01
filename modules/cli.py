"""
CLI argument parsing and interactive mode functionality.
"""

import argparse
import getpass
import sys
from .config import BenchmarkConfig
from .utils import resolve_env_var
from .benchmark import TransformationBenchmark


def interactive_mode():
    """Run benchmark in interactive mode"""
    
    print("🤖 Transformation Benchmark - Interactive Mode")
    print("=" * 50)
    
    config = BenchmarkConfig()
    
    print("\n📋 Configuration Setup")
    print("-" * 25)
    
    # Model configuration
    print("\n🤖 Model Configuration:")
    print("💡 Tip: Use 'env:VARIABLE_NAME' to reference environment variables")
    print("📝 Example: env:OPENROUTER_API_KEY")
    
    base_url_input = input(f"Base URL [{config.base_url}]: ").strip() or config.base_url
    try:
        config.base_url = resolve_env_var(base_url_input)
        if base_url_input.startswith('env:'):
            print(f"✅ Resolved environment variable to: {config.base_url}")
    except ValueError as e:
        print(f"❌ Error: {e}")
        config.base_url = base_url_input
    
    api_key_input = getpass.getpass(f"API Key [{'***masked***' if config.api_key else 'none'}]: ").strip()
    if api_key_input:
        try:
            config.api_key = resolve_env_var(api_key_input)
            if api_key_input.startswith('env:'):
                print(f"✅ Resolved environment variable for API key")
        except ValueError as e:
            print(f"❌ Error: {e}")
            config.api_key = api_key_input
    
    config.model_name = input("Model name (leave empty for local): ").strip()
    
    # Benchmark configuration
    print("\n📊 Benchmark Configuration:")
    try:
        config.max_complexity = max(1, min(5, int(input(f"Max complexity level (1-5) [{config.max_complexity}]: ") or config.max_complexity)))
    except ValueError:
        print("Invalid input, using default")
    
    try:
        config.trials_per_complexity = max(1, int(input(f"Trials per complexity [{config.trials_per_complexity}]: ") or config.trials_per_complexity))
    except ValueError:
        print("Invalid input, using default")
    
    # Content types
    print(f"\nAvailable content types: {', '.join(config.content_types)}")
    content_input = input("Content types (comma-separated, or press Enter for all): ").strip()
    if content_input:
        config.content_types = [t.strip() for t in content_input.split(',') if t.strip()]
    
    # Topic
    config.topic = input("Topic for content generation (optional): ").strip() or None
    
    # Temperature
    try:
        config.temperature = max(0.0, min(2.0, float(input(f"Temperature [{config.temperature}]: ") or config.temperature)))
    except ValueError:
        print("Invalid input, using default")
    
    # Show configuration summary
    total_trials = config.max_complexity * config.trials_per_complexity
    print(f"\n🚀 Starting benchmark with:")
    print(f"  Model: {config.model_name or 'Local/Default'}")
    print(f"  Complexity: 1-{config.max_complexity}")
    print(f"  Trials: {config.trials_per_complexity} per level")
    print(f"  Total trials: {total_trials}")
    print(f"  Content: {', '.join(config.content_types)}")
    if config.topic:
        print(f"  Topic: {config.topic}")
    
    if input("\nProceed? (y/N): ").strip().lower() != 'y':
        print("❌ Benchmark cancelled")
        return
    
    # Run benchmark
    try:
        benchmark = TransformationBenchmark(config)
        summary = benchmark.run_benchmark()
        benchmark.save_detailed_results()
        return summary
    except Exception as e:
        print(f"❌ Benchmark setup failed: {e}")
        return None


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    
    parser = argparse.ArgumentParser(
        description="Enhanced Self-Evaluating Transformation Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with local model
  python benchmark.py --quick
  
  # Full benchmark with specific model
  python benchmark.py --model gpt-3.5-turbo --url https://api.openai.com/v1
  
  # Using OpenRouter with environment variables
  python benchmark.py --api-key env:OPENROUTER_API_KEY --url env:OPENROUTER_BASE_URL --model anthropic/claude-3-sonnet
  
  # Using environment variables (set first: export OPENROUTER_API_KEY=your_key)
  python benchmark.py --api-key env:OPENROUTER_API_KEY --url https://openrouter.ai/api/v1
  
  # Topic-focused benchmark with logging
  python benchmark.py --topic "blockchain" --content code,text --log-file blockchain_test.log
  
  # Interactive mode
  python benchmark.py --interactive

Environment Variables:
  You can reference environment variables using the 'env:' prefix:
  --api-key env:OPENROUTER_API_KEY    # Uses $OPENROUTER_API_KEY
  --url env:OPENROUTER_BASE_URL       # Uses $OPENROUTER_BASE_URL
  --api-key env:OPENAI_API_KEY        # Uses $OPENAI_API_KEY
        """
    )
    
    # Mode selection
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--quick', '-q', action='store_true', help='Quick benchmark (2 complexity levels, 1 trial each)')
    
    # Model configuration
    parser.add_argument('--url', '--base-url', default="http://localhost:1234/v1", 
                       help='Base URL for the LLM API (supports env:VARIABLE_NAME format)')
    parser.add_argument('--api-key', default="not-needed", 
                       help='API key for the LLM service (supports env:VARIABLE_NAME format, e.g., env:OPENROUTER_API_KEY)')
    parser.add_argument('--model', '--model-name', default="", help='Model name (leave empty for local models)')
    parser.add_argument('--creative-model', default="", help='Specific model for creative content generation')
    parser.add_argument('--structured-model', default="", help='Specific model for structured tasks')
    parser.add_argument('--transform-model', default="", help='Specific model for transformations')
    
    # Benchmark configuration
    parser.add_argument('--complexity', '--max-complexity', type=int, default=5, help='Maximum complexity level (1-5)')
    parser.add_argument('--trials', '--trials-per-complexity', type=int, default=3, help='Number of trials per complexity level')
    parser.add_argument('--temperature', type=float, default=0.3, help='Base temperature for LLM')
    
    # Content configuration
    parser.add_argument('--content', '--content-types', default="code,text,data,configuration,documentation", help='Comma-separated list of content types')
    parser.add_argument('--topic', help='Topic for content generation (makes content topic-specific)')
    
    # Output configuration
    parser.add_argument('--output', '-o', default=None, help='Output filename for results (auto-generated if not specified)')
    parser.add_argument('--log-file', default=None, help='Log filename (auto-generated if not specified)')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    return parser
