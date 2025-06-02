"""
CLI argument parsing and interactive mode functionality with enhanced content type support.
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
    config.creative_api_key = resolve_arg(args.creative_api_key) if args.creative_api_key else config.api_key
    config.verification_api_key = resolve_arg(args.verification_api_key) if args.verification_api_key else config.api_key
    config.transform_api_key = resolve_arg(args.transform_api_key) if args.transform_api_key else config.api_key
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
    """Run benchmark in interactive mode with enhanced content type selection"""
    
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
    api_key_input = getpass.getpass(f"Default API Key [{'***masked***' if config.api_key else 'none'}]: ").strip()
    if api_key_input:
        try:
            config.api_key = resolve_env_var(api_key_input)
            if api_key_input.startswith('env:'):
                print(f"✅ Resolved environment variable for API key")
        except ValueError as e:
            print(f"❌ Error: {e}")
            config.api_key = api_key_input
    
    # Ask if user wants different API keys for different stages
    use_different_api_keys = input("Use different API keys for different stages? (y/N): ").strip().lower() == 'y'
    if use_different_api_keys:
        creative_api_key_input = getpass.getpass(f"Creative API Key [{'***masked***' if config.api_key else 'none'}]: ").strip()
        if creative_api_key_input:
            try:
                config.creative_api_key = resolve_env_var(creative_api_key_input)
                if creative_api_key_input.startswith('env:'):
                    print(f"✅ Resolved environment variable for creative API key")
            except ValueError as e:
                print(f"❌ Error: {e}")
                config.creative_api_key = creative_api_key_input
        
        verification_api_key_input = getpass.getpass(f"Verification API Key [{'***masked***' if config.api_key else 'none'}]: ").strip()
        if verification_api_key_input:
            try:
                config.verification_api_key = resolve_env_var(verification_api_key_input)
                if verification_api_key_input.startswith('env:'):
                    print(f"✅ Resolved environment variable for verification API key")
            except ValueError as e:
                print(f"❌ Error: {e}")
                config.verification_api_key = verification_api_key_input
        
        transform_api_key_input = getpass.getpass(f"Transform API Key [{'***masked***' if config.api_key else 'none'}]: ").strip()
        if transform_api_key_input:
            try:
                config.transform_api_key = resolve_env_var(transform_api_key_input)
                if transform_api_key_input.startswith('env:'):
                    print(f"✅ Resolved environment variable for transform API key")
            except ValueError as e:
                print(f"❌ Error: {e}")
                config.transform_api_key = transform_api_key_input
    
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
    
    # Enhanced content types selection
    print(f"\n📝 Content Type Selection:")
    print("Available content types:")
    print("  🎨 Creative: poetry, short_story, song_lyrics, screenplay, philosophical_essay")
    print("  💻 Technical: code, api_documentation, database_schema, system_logs, scientific_paper")
    print("  🔀 Hybrid: interview_transcript, product_review, tutorial, case_study")
    print("  📄 Traditional: text, data, configuration, documentation")
    print(f"\nDefault selection: {', '.join(config.content_types)}")
    
    content_input = input("Content types (comma-separated, or press Enter for default): ").strip()
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
    
    # Show API key information if different keys are used
    if use_different_api_keys and any([
        config.creative_api_key != config.api_key,
        config.verification_api_key != config.api_key,
        config.transform_api_key != config.api_key
    ]):
        print(f"  API Keys:")
        print(f"    Default: {'***masked***' if config.api_key else 'none'}")
        if config.creative_api_key != config.api_key:
            print(f"    Creative: {'***masked***' if config.creative_api_key else 'none'}")
        if config.verification_api_key != config.api_key:
            print(f"    Verification: {'***masked***' if config.verification_api_key else 'none'}")
        if config.transform_api_key != config.api_key:
            print(f"    Transform: {'***masked***' if config.transform_api_key else 'none'}")
    
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
    """Create command line argument parser with enhanced content type support"""
    
    parser = argparse.ArgumentParser(
        description="Enhanced Self-Evaluating Transformation Benchmark with Exponential Complexity Scaling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Content Types:
  Creative: poetry, short_story, song_lyrics, screenplay, philosophical_essay
  Technical: code, api_documentation, database_schema, system_logs, scientific_paper  
  Hybrid: interview_transcript, product_review, tutorial, case_study
  Traditional: text, data, configuration, documentation

Complexity Levels:
  Level 1: Simple structural transformations (format changes)
  Level 2: Enhanced structural + basic conditions  
  Level 3: Cross-format + semantic transformations
  Level 4: Complex multi-stage transformations
  Level 5: Creative reinterpretation + meta-transformation

Examples:
  # Quick test with creative content types
  python benchmark.py --quick --content "poetry,short_story,code"
  
  # Full benchmark with diverse content mix
  python benchmark.py --content "code,poetry,tutorial,api_documentation" --complexity 4
  
  # Topic-focused creative benchmark
  python benchmark.py --topic "artificial intelligence" --content "poetry,philosophical_essay,tutorial" 
  
  # Using OpenRouter with specific models for different stages
  python benchmark.py --api-key env:OPENROUTER_API_KEY --url env:OPENROUTER_BASE_URL \\
    --creative-model anthropic/claude-3-sonnet --verification-model anthropic/claude-3-haiku
  
  # Using different API keys and URLs for different stages
  python benchmark.py --creative-model qwen3-0.6b --transform-model qwen3-0.6b \\
    --verification-model deepseek/deepseek-r1-0528-qwen3-8b:free \\
    --verification-api-key env:OPENROUTER_API_KEY --verification-url env:OPENROUTER_BASE_URL
  
  # High complexity test with multiple verification attempts
  python benchmark.py --complexity 5 --verification-attempts 3 --verification-aggregation best
  
  # Interactive mode for guided setup
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
                       help='Default API key for the LLM service (supports env:VARIABLE_NAME format, e.g., env:OPENROUTER_API_KEY)')
    parser.add_argument('--creative-api-key', default="", 
                       help='API key for creative LLM (defaults to --api-key if not specified)')
    parser.add_argument('--verification-api-key', default="", 
                       help='API key for verification LLM (defaults to --api-key if not specified)')
    parser.add_argument('--transform-api-key', default="", 
                       help='API key for transform LLM (defaults to --api-key if not specified)')
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
    
    # Content configuration with enhanced default
    default_content = "code,text,data,poetry,short_story,tutorial,api_documentation"
    parser.add_argument('--content', '--content-types', default=default_content, 
                       help=f'Comma-separated list of content types (default: {default_content})')
    parser.add_argument('--topic', help='Topic for content generation (makes content topic-specific)')
    
    # Output configuration
    parser.add_argument('--output', '-o', default=None, help='Output filename for results (auto-generated if not specified)')
    parser.add_argument('--log-file', default=None, help='Log filename (auto-generated if not specified)')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    return parser