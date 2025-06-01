#!/usr/bin/env python3
"""
Self-Evaluating Transformation Benchmark

Main entry point for the benchmark application.
"""

import sys
from dotenv import load_dotenv
from modules import BenchmarkConfig, TransformationBenchmark, create_parser, interactive_mode, parse_args_to_config
from modules.utils import resolve_env_var

# Load environment variables from .env file
load_dotenv()


def main():
    """Main execution function"""
    
    parser = create_parser()
    args = parser.parse_args()
    
    if args.interactive:
        return interactive_mode()
    
    # Use parse_args_to_config to create the configuration
    try:
        config = parse_args_to_config(args)
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("💡 Tip: Use 'env:VARIABLE_NAME' to reference environment variables")
        print("📝 Example: --api-key env:OPENROUTER_API_KEY --base-url env:OPENROUTER_BASE_URL")
        return 1
    
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