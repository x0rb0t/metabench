"""
CLI argument parsing and interactive mode functionality.
"""

import argparse
import getpass
import sys
from .config import BenchmarkConfig
from .utils import resolve_env_var
from .benchmark import TransformationBenchmark


def parse_args_to_config(args) -> BenchmarkConfig:
    """Convert parsed arguments to BenchmarkConfig"""
    
    # Handle environment variable resolution
    def resolve_arg(value):
        try:
            return resolve_env_var(value)
        except ValueError as e:
            print(f"Warning: {e}")
            return value
    
    config = BenchmarkConfig()
    
    # Base URLs
    config.base_url = resolve_arg(args.url)
    config.creative_base_url = resolve_arg(args.creative_url) if args.creative_url else config.base_url
    config.verification_base_url = resolve_arg(args.verification_url) if args.verification_url else config.base_url
    config.transform_base_url = resolve_arg(args.transform_url) if args.transform_url else config.base_url
    
    # API and models
    config.api_key = resolve_arg(args.api_key)
    config.model_name = args.model
    config.creative_model = args.creative_model
    config.verification_model = args.verification_model
    config.transform_model = args.transform_model
    
    # Benchmark settings
    config.max_complexity = args.complexity
    config.trials_per_complexity = args.trials
    config.temperature = args.temperature
    
    # Individual temperature settings (with smart defaults)
    config.creative_temperature = args.creative_temperature if args.creative_temperature is not None else 0.7
    config.verification_temperature = args.verification_temperature if args.verification_temperature is not None else 0.1
    config.transform_temperature = args.transform_temperature if args.transform_temperature is not None else args.temperature
    
    # Retry settings
    config.max_retries = args.max_retries
    
    # Verification settings
    config.verification_attempts = args.verification_attempts
    config.verification_aggregation = args.verification_aggregation
    
    # Content settings
    config.content_types = [t.strip() for t in args.content.split(',') if t.strip()]
    config.topic = args.topic
    
    # Output settings
    config.log_file = args.log_file
    
    # Quick mode override
    if args.quick:
        config.max_complexity = 2
        config.trials_per_complexity = 1
        config.verification_attempts = 1
        print("🚀 Quick mode: 2 complexity levels, 1 trial each")
    
    return config


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
    
    # Base URLs configuration
    print("\n🌐 Base URL Configuration:")
    base_url_input = input(f"Default Base URL [{config.base_url}]: ").strip() or config.base_url
    try:
        config.base_url = resolve_env_var(base_url_input)
        if base_url_input.startswith('env:'):
            print(f"✅ Resolved environment variable to: {config.base_url}")
    except ValueError as e:
        print(f"❌ Error: {e}")
        config.base_url = base_url_input
    
    # Ask if user wants different URLs for different stages
    use_different_urls = input("Use different base URLs for different stages? (y/N): ").strip().lower() == 'y'
    if use_different_urls:
        creative_url_input = input(f"Creative LLM Base URL [{config.base_url}]: ").strip()
        if creative_url_input:
            try:
                config.creative_base_url = resolve_env_var(creative_url_input)
            except ValueError as e:
                print(f"❌ Error: {e}")
                config.creative_base_url = creative_url_input
        
        verification_url_input = input(f"Verification LLM Base URL [{config.base_url}]: ").strip()
        if verification_url_input:
            try:
                config.verification_base_url = resolve_env_var(verification_url_input)
            except ValueError as e:
                print(f"❌ Error: {e}")
                config.verification_base_url = verification_url_input
        
        transform_url_input = input(f"Transform LLM Base URL [{config.base_url}]: ").strip()
        if transform_url_input:
            try:
                config.transform_base_url = resolve_env_var(transform_url_input)
            except ValueError as e:
                print(f"❌ Error: {e}")
                config.transform_base_url = transform_url_input
    
    # API Key
    api_key_input = getpass.getpass(f"API Key [{'***masked***' if config.api_key else 'none'}]: ").strip()
    if api_key_input:
        try:
            config.api_key = resolve_env_var(api_key_input)
            if api_key_input.startswith('env:'):
                print(f"✅ Resolved environment variable for API key")
        except ValueError as e:
            print(f"❌ Error: {e}")
            config.api_key = api_key_input
    
    # Model names
    config.model_name = input("Default model name (leave empty for local): ").strip()
    
    use_different_models = input("Use different models for different stages? (y/N): ").strip().lower() == 'y'
    if use_different_models:
        config.creative_model = input(f"Creative model [{config.model_name}]: ").strip()
        config.verification_model = input(f"Verification model [{config.model_name}]: ").strip()
        config.transform_model = input(f"Transform model [{config.model_name}]: ").strip()
    
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
        config.temperature = max(0.0, min(2.0, float(input(f"Base temperature [{config.temperature}]: ") or config.temperature)))
        config.creative_temperature = max(0.0, min(2.0, float(input(f"Creative temperature [{config.creative_temperature}]: ") or config.creative_temperature)))
        config.verification_temperature = max(0.0, min(2.0, float(input(f"Verification temperature [{config.verification_temperature}]: ") or config.verification_temperature)))
        config.transform_temperature = max(0.0, min(2.0, float(input(f"Transform temperature [{config.transform_temperature}]: ") or config.transform_temperature)))
    except ValueError:
        print("Invalid input, using defaults")
    
    # Retry configuration
    print("\n🔄 Retry Configuration:")
    try:
        config.max_retries = max(1, int(input(f"Max retries per operation [{config.max_retries}]: ") or config.max_retries))
    except ValueError:
        print("Invalid input, using default")
    
    # Verification configuration
    print("\n🔍 Verification Configuration:")
    try:
        config.verification_attempts = max(1, int(input(f"Verification attempts per trial [{config.verification_attempts}]: ") or config.verification_attempts))
    except ValueError:
        print("Invalid input, using default")
    
    if config.verification_attempts > 1:
        print("Available aggregation methods: best, avg, worst")
        agg_input = input(f"Verification aggregation method [{config.verification_aggregation}]: ").strip().lower()
        if agg_input in ['best', 'avg', 'worst']:
            config.verification_aggregation = agg_input
    
    # Show configuration summary
    total_trials = config.max_complexity * config.trials_per_complexity
    print(f"\n🚀 Starting benchmark with:")
    print(f"  Model: {config.model_name or 'Local/Default'}")
    if use_different_models and any([config.creative_model, config.verification_model, config.transform_model]):
        print(f"    Creative: {config.creative_model or config.model_name or 'Local/Default'}")
        print(f"    Verification: {config.verification_model or config.model_name or 'Local/Default'}")
        print(f"    Transform: {config.transform_model or config.model_name or 'Local/Default'}")
    
    print(f"  Base URLs:")
    print(f"    Default: {config.base_url}")
    if use_different_urls:
        print(f"    Creative: {config.creative_base_url}")
        print(f"    Verification: {config.verification_base_url}")
        print(f"    Transform: {config.transform_base_url}")
    
    print(f"  Complexity: 1-{config.max_complexity}")
    print(f"  Trials: {config.trials_per_complexity} per level")
    print(f"  Total trials: {total_trials}")
    print(f"  Content: {', '.join(config.content_types)}")
    if config.topic:
        print(f"  Topic: {config.topic}")
    print(f"  Max retries: {config.max_retries}")
    print(f"  Verification: {config.verification_attempts} attempts ({config.verification_aggregation})")
    
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
        description="Enhanced Self-Evaluating Transformation Benchmark with Retry Logic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with local model
  python benchmark.py --quick
  
  # Full benchmark with specific model
  python benchmark.py --model gpt-3.5-turbo --url https://api.openai.com/v1
  
  # Using OpenRouter with environment variables
  python benchmark.py --api-key env:OPENROUTER_API_KEY --url env:OPENROUTER_BASE_URL --model anthropic/claude-3-sonnet
  
  # Different URLs for different stages
  python benchmark.py --url http://localhost:1234/v1 --creative-url http://creative.local:1234/v1 --verification-url http://verification.local:1234/v1
  
  # With retry and verification configuration
  python benchmark.py --max-retries 5 --verification-attempts 3 --verification-aggregation best
  
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
                       help='Default base URL for the LLM API (supports env:VARIABLE_NAME format)')
    parser.add_argument('--creative-url', default="", 
                       help='Base URL for creative LLM (defaults to --url if not specified)')
    parser.add_argument('--verification-url', default="", 
                       help='Base URL for verification LLM (defaults to --url if not specified)')
    parser.add_argument('--transform-url', default="", 
                       help='Base URL for transform LLM (defaults to --url if not specified)')
    
    parser.add_argument('--api-key', default="not-needed", 
                       help='API key for the LLM service (supports env:VARIABLE_NAME format, e.g., env:OPENROUTER_API_KEY)')
    parser.add_argument('--model', '--model-name', default="", help='Default model name (leave empty for local models)')
    parser.add_argument('--creative-model', default="", help='Specific model for creative content generation')
    parser.add_argument('--verification-model', default="", help='Specific model for verification tasks')
    parser.add_argument('--transform-model', default="", help='Specific model for transformations')
    
    # Benchmark configuration
    parser.add_argument('--complexity', '--max-complexity', type=int, default=5, help='Maximum complexity level (1-5)')
    parser.add_argument('--trials', '--trials-per-complexity', type=int, default=3, help='Number of trials per complexity level')
    parser.add_argument('--temperature', type=float, default=0.3, help='Base temperature for LLM')
    parser.add_argument('--creative-temperature', type=float, help='Temperature for creative content generation (defaults to 0.7)')
    parser.add_argument('--verification-temperature', type=float, help='Temperature for verification tasks (defaults to 0.1)')
    parser.add_argument('--transform-temperature', type=float, help='Temperature for transformation tasks (defaults to --temperature)')
    
    # Retry configuration
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum retries per operation (default: 3)')
    
    # Verification configuration
    parser.add_argument('--verification-attempts', type=int, default=1, help='Number of verification attempts per trial (default: 1)')
    parser.add_argument('--verification-aggregation', choices=['best', 'avg', 'worst'], default='avg', 
                       help='How to aggregate multiple verification scores (default: avg)')
    
    # Content configuration
    parser.add_argument('--content', '--content-types', default="code,text,data,configuration,documentation", help='Comma-separated list of content types')
    parser.add_argument('--topic', help='Topic for content generation (makes content topic-specific)')
    
    # Output configuration
    parser.add_argument('--output', '-o', default=None, help='Output filename for results (auto-generated if not specified)')
    parser.add_argument('--log-file', default=None, help='Log filename (auto-generated if not specified)')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    return parser