#!/usr/bin/env python3
"""
Self-Evaluating Transformation Benchmark

Main entry point for the benchmark application.
"""

import sys
from dotenv import load_dotenv
from modules import BenchmarkConfig, TransformationBenchmark, create_parser, interactive_mode
from modules.utils import resolve_env_var

# Load environment variables from .env file
load_dotenv()


def main():
    """Main execution function"""
    
    parser = create_parser()
    args = parser.parse_args()
    
    if args.interactive:
        return interactive_mode()
    
    # Resolve environment variables for API key and base URL
    try:
        resolved_api_key = resolve_env_var(args.api_key)
        resolved_base_url = resolve_env_var(args.url)
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("💡 Tip: Use 'env:VARIABLE_NAME' to reference environment variables")
        print("📝 Example: --api-key env:OPENROUTER_API_KEY --url env:OPENROUTER_BASE_URL")
        return 1
    
    # Create configuration
    config = BenchmarkConfig(
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        model_name=args.model,
        creative_model=args.creative_model,
        structured_model=args.structured_model,
        transform_model=args.transform_model,
        temperature=args.temperature,
        topic=args.topic,
        content_types=[t.strip() for t in args.content.split(',') if t.strip()],
        log_file=args.log_file
    )
    
    if args.quick:
        config.max_complexity = 2
        config.trials_per_complexity = 1
        config.content_types = ["code", "text"]
        print("🏃‍♂️ Quick mode: 2 complexity levels, 1 trial each, code+text only")
    else:
        config.max_complexity = max(1, min(5, args.complexity))
        config.trials_per_complexity = max(1, args.trials)
    
    try:
        benchmark = TransformationBenchmark(config)
        summary = benchmark.run_benchmark()
        benchmark.save_detailed_results(args.output)
        
        print(f"\n🎉 Benchmark completed successfully!")
        return summary
        
    except KeyboardInterrupt:
        print("\n⏹️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()