"""
Main benchmark runner class.
"""

import json
import time
import random
import statistics
import logging
import os
from datetime import datetime
from typing import List
from dataclasses import asdict
from .config import BenchmarkConfig, BenchmarkResult, BenchmarkSummary, sanitize_config_for_logging, sanitize_config_for_saving, validate_config
from .utils import setup_logging
from .llm import BenchmarkLLM
from .generators import EnhancedInstructionGenerator, EnhancedContentGenerator
from .engines import TransformationEngine, EnhancedVerificationEngine


class TransformationBenchmark:
    """Main benchmark runner with enhanced logging, progress tracking, retry logic, and multi-verification scoring"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.file_logger, self.console_logger = setup_logging(config.log_file)
        self.file_logger.info("Initializing Transformation Benchmark")
        self.file_logger.info(f"Configuration: {sanitize_config_for_logging(config)}")
        
        # Validate configuration
        config_issues = validate_config(config)
        if config_issues:
            error_msg = f"Configuration validation failed:\n" + "\n".join([f"  - {issue}" for issue in config_issues])
            self.file_logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.llm = BenchmarkLLM(config, self.file_logger)
        self.instruction_generator = EnhancedInstructionGenerator(self.file_logger)
        self.content_generator = EnhancedContentGenerator(self.llm, config.topic, self.file_logger)
        self.transformation_engine = TransformationEngine(self.llm, self.file_logger)
        self.verification_engine = EnhancedVerificationEngine(self.llm, self.file_logger)
        self.results: List[BenchmarkResult] = []
        
        # Progress tracking
        self.total_trials = self.config.max_complexity * self.config.trials_per_complexity
        self.current_trial = 0
        self.benchmark_start_time = None
        
        # Log new configuration details
        self.file_logger.info(f"Retry configuration: max_retries={config.max_retries}")
        self.file_logger.info(f"Verification configuration: attempts={config.verification_attempts}, aggregation={config.verification_aggregation}")
        
        llm_info = self.llm.get_llm_info()
        for llm_type, info in llm_info.items():
            self.file_logger.info(f"{llm_type.capitalize()} LLM: {info['base_url']} (temp={info['temperature']})")
    
    def _log_content_safely(self, logger: logging.Logger, content: str, label: str, max_chars: int = 1000):
        """Log content with intelligent truncation and optional full content saving"""
        if len(content) <= max_chars:
            logger.info(f"{label} START:")
            logger.info(content)
            logger.info(f"{label} END")
        else:
            logger.info(f"{label} START (truncated - full length: {len(content)} chars):")
            logger.info(content[:max_chars//2] + "\n... [TRUNCATED] ...\n" + content[-max_chars//2:])
            logger.info(f"{label} END")
            
    
    def _update_progress(self, complexity_level: int, trial: int, content_type: str):
        """Update and display progress information"""
        self.current_trial += 1
        progress_pct = (self.current_trial / self.total_trials) * 100
        
        print(f"\n  Trial {self.current_trial}/{self.total_trials} ({progress_pct:.1f}%) - {content_type}")
        
        # Log progress to file
        self.file_logger.info(f"Progress: Trial {self.current_trial}/{self.total_trials} ({progress_pct:.1f}%) - "
                             f"Complexity {complexity_level}, Type {content_type}")
    
    def run_single_trial(self, complexity_level: int, content_type: str) -> BenchmarkResult:
        """Run a single benchmark trial with comprehensive logging, progress tracking, and retry logic"""
        
        start_time = time.time()
        errors = []
        
        # Initialize attempt counters
        content_attempts = 0
        instruction_attempts = 0
        transformation_attempts = 0
        verification_attempts = 0
        verification_scores = []
        
        # Enhanced trial start logging for file only
        self.file_logger.info("=" * 50)
        self.file_logger.info(f"TRIAL START: Complexity Level {complexity_level}, Content Type: {content_type}")
        self.file_logger.info(f"Trial {self.current_trial + 1}/{self.total_trials}")
        self.file_logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.file_logger.info(f"Max retries: {self.config.max_retries}")
        self.file_logger.info(f"Verification attempts: {self.config.verification_attempts} ({self.config.verification_aggregation})")
        self.file_logger.info("=" * 50)
        
        print(f"    🚀 Starting trial: Complexity {complexity_level}, Type {content_type}")
        
        try:
            # Phase 1: Generate content
            print(f"  📝 Phase 1/4: Generating {content_type} content...")
            
            self.file_logger.info("-" * 30)
            self.file_logger.info(f"PHASE 1 START: Content Generation ({content_type})")
            self.file_logger.info("-" * 30)
            
            generated_content, content_variations, content_attempts = self.content_generator.generate_content(
                content_type, self.config.max_retries
            )
            
            self.file_logger.info(f"Generated content length: {len(generated_content)} characters")
            self.file_logger.info(f"Content variations applied: {content_variations}")
            self.file_logger.info(f"Content generation attempts: {content_attempts}")
            self._log_content_safely(self.file_logger, generated_content, "GENERATED CONTENT")
            self.file_logger.info(f"PHASE 1 COMPLETE: Content Generation")
            self.file_logger.info("-" * 30)
            
            # Phase 2: Generate transformation instruction
            print(f"  🎯 Phase 2/4: Creating complexity {complexity_level} transformation...")
            
            self.file_logger.info("-" * 30)
            self.file_logger.info(f"PHASE 2 START: Instruction Generation (Complexity {complexity_level})")
            self.file_logger.info("-" * 30)
            
            instruction, instruction_attempts = self.instruction_generator.generate_instruction(
                complexity_level, content_type, self.config.max_retries
            )
            
            self.file_logger.info("INSTRUCTION START:")
            self.file_logger.info(json.dumps(instruction, indent=2))
            self.file_logger.info("INSTRUCTION END")
            self.file_logger.info(f"Instruction generation attempts: {instruction_attempts}")
            self.file_logger.info(f"PHASE 2 COMPLETE: Instruction Generation")
            self.file_logger.info("-" * 30)
            
            # Phase 3: Apply transformation
            print(f"  ⚙️  Phase 3/4: Applying transformation...")
            
            self.file_logger.info("-" * 30)
            self.file_logger.info("PHASE 3 START: Transformation Application")
            self.file_logger.info("-" * 30)
            self.file_logger.info(f"Input content length: {len(generated_content)} characters")
            
            transformed_content, transformation_attempts = self.transformation_engine.apply_transformation(
                generated_content, instruction, self.config.max_retries
            )
            
            self.file_logger.info(f"Output content length: {len(transformed_content)} characters")
            self.file_logger.info(f"Transformation attempts: {transformation_attempts}")
            self._log_content_safely(self.file_logger, transformed_content, "TRANSFORMED CONTENT")
            self.file_logger.info("PHASE 3 COMPLETE: Transformation Application")
            self.file_logger.info("-" * 30)
            
            # Phase 4: Verify results
            print(f"  🔍 Phase 4/4: Verifying results...")
            
            self.file_logger.info("-" * 30)
            self.file_logger.info("PHASE 4 START: Verification")
            self.file_logger.info("-" * 30)
            
            quality_score, completion_rate, specific_scores, verification_attempts, verification_scores = self.verification_engine.verify_transformation(
                generated_content, instruction, transformed_content,
                self.config.max_retries,
                self.config.verification_attempts,
                self.config.verification_aggregation
            )
            
            self.file_logger.info(f"Quality Score: {quality_score:.2f}/10")
            self.file_logger.info(f"Completion Rate: {completion_rate:.2%}")
            self.file_logger.info(f"Specific Scores: {specific_scores}")
            self.file_logger.info(f"Verification attempts: {verification_attempts}")
            self.file_logger.info(f"All verification scores: {verification_scores}")
            self.file_logger.info("PHASE 4 COMPLETE: Verification")
            self.file_logger.info("-" * 30)
            
        except (ConnectionError, TimeoutError) as e:
            error_msg = f"Network error: {str(e)}"
            self.file_logger.error(f"TRIAL FAILURE: {error_msg}")
            print(f"    🌐 {error_msg}")
            errors.append(error_msg)
            generated_content, content_variations, instruction, transformed_content = "", [], {}, ""
            quality_score, completion_rate, specific_scores = 0.0, 0.0, {}
            verification_scores = [0.0]
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing error: {str(e)}"
            self.file_logger.error(f"TRIAL FAILURE: {error_msg}")
            print(f"    📄 {error_msg}")
            errors.append(error_msg)
            generated_content, content_variations, instruction, transformed_content = "", [], {}, ""
            quality_score, completion_rate, specific_scores = 0.0, 0.0, {}
            verification_scores = [0.0]
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.file_logger.error("-" * 30)
            self.file_logger.error("TRIAL FAILURE")
            self.file_logger.error("-" * 30)
            self.file_logger.error(f"Error occurred: {str(e)}", exc_info=True)
            self.file_logger.error("-" * 30)
            
            print(f"    ❌ {error_msg}")
            errors.append(error_msg)
            generated_content, content_variations, instruction, transformed_content = "", [], {}, ""
            quality_score, completion_rate, specific_scores = 0.0, 0.0, {}
            verification_scores = [0.0]
        
        execution_time = time.time() - start_time
        
        # Enhanced trial completion logging for file only
        self.file_logger.info("=" * 50)
        self.file_logger.info(f"TRIAL COMPLETE: Execution time {execution_time:.2f}s")
        self.file_logger.info(f"Final Results - Quality: {quality_score:.2f}/10, Completion: {completion_rate:.2%}")
        self.file_logger.info(f"Attempt counts - Content: {content_attempts}, Instruction: {instruction_attempts}, "
                             f"Transform: {transformation_attempts}, Verify: {verification_attempts}")
        if errors:
            self.file_logger.info(f"Errors encountered: {len(errors)}")
        self.file_logger.info("=" * 50)
        
        print(f"    ⏱️  Trial completed in {execution_time:.2f}s")
        if content_attempts + instruction_attempts + transformation_attempts + verification_attempts > 4:
            total_attempts = content_attempts + instruction_attempts + transformation_attempts + verification_attempts
            print(f"    🔄 Total attempts across all phases: {total_attempts}")
        
        return BenchmarkResult(
            complexity_level=complexity_level,
            content_type=content_type,
            generated_content=generated_content,
            transformation_instruction=json.dumps(instruction, indent=2),
            transformed_content=transformed_content,
            verification_score=quality_score,
            instruction_completion_rate=completion_rate,
            execution_time=execution_time,
            errors=errors,
            content_variations=content_variations,
            specific_scores=specific_scores,
            content_generation_attempts=content_attempts,
            instruction_generation_attempts=instruction_attempts,
            transformation_attempts=transformation_attempts,
            verification_attempts=verification_attempts,
            verification_scores=verification_scores
        )
    
    def run_benchmark(self) -> BenchmarkSummary:
        """Run the complete benchmark suite with progress tracking"""
        
        self.benchmark_start_time = time.time()
        self.file_logger.info("Starting benchmark suite")
        
        print("🚀 Starting Transformation Benchmark")
        print(f"Configuration: {self.config.max_complexity} complexity levels, "
              f"{self.config.trials_per_complexity} trials each")
        print(f"Content types: {', '.join(self.config.content_types)}")
        if self.config.topic:
            print(f"Topic: {self.config.topic}")
        
        # Display LLM configuration
        llm_info = self.llm.get_llm_info()
        print(f"LLM Configuration:")
        for llm_type, info in llm_info.items():
            model_name = info['model'] or 'Local/Default'
            print(f"  {llm_type.capitalize()}: {info['base_url']} (model={model_name}, temp={info['temperature']})")
        
        print(f"Retry settings: max_retries={self.config.max_retries}")
        print(f"Verification: {self.config.verification_attempts} attempts, {self.config.verification_aggregation} aggregation")
        print(f"Total trials: {self.total_trials}")
        print(f"Logging to: {self.file_logger.handlers[0].baseFilename if self.file_logger.handlers else 'console only'}")
        print("-" * 60)
        
        for complexity in range(1, self.config.max_complexity + 1):
            print(f"\n📊 Complexity Level {complexity}")
            self.file_logger.info(f"Starting complexity level {complexity}")
            
            for trial in range(self.config.trials_per_complexity):
                content_type = random.choice(self.config.content_types)
                
                # Update progress before starting trial
                self._update_progress(complexity, trial + 1, content_type)
                
                result = self.run_single_trial(complexity, content_type)
                self.results.append(result)
                
                # Log detailed results
                self.file_logger.info(f"Trial result: Score={result.verification_score:.2f}, "
                               f"Completion={result.instruction_completion_rate:.2%}, "
                               f"Time={result.execution_time:.2f}s")
                
                # Print results
                status_emoji = "✅" if result.verification_score > 5 else "⚠️" if result.verification_score > 2 else "❌"
                print(f"    {status_emoji} Results:")
                print(f"      📊 Quality Score: {result.verification_score:.2f}/10")
                print(f"      ✅ Completion Rate: {result.instruction_completion_rate:.1%}")
                print(f"      ⏱️  Execution Time: {result.execution_time:.2f}s")
                print(f"      📝 Content Length: {len(result.generated_content)} → {len(result.transformed_content)} chars")
                print(f"      🎨 Variations: {', '.join(result.content_variations)}")
                
                # Show attempt counts if any retries occurred
                total_attempts = (result.content_generation_attempts + result.instruction_generation_attempts + 
                                result.transformation_attempts + result.verification_attempts)
                if total_attempts > 4:  # More than 1 attempt per phase
                    print(f"      🔄 Attempts: Content={result.content_generation_attempts}, "
                          f"Instruction={result.instruction_generation_attempts}, "
                          f"Transform={result.transformation_attempts}, "
                          f"Verify={result.verification_attempts}")
                
                # Show verification score range if multiple verification attempts
                if len(result.verification_scores) > 1:
                    score_range = f"{min(result.verification_scores):.2f}-{max(result.verification_scores):.2f}"
                    print(f"      🎯 Verification Range: {score_range} ({self.config.verification_aggregation})")
                
                if result.errors:
                    print(f"      ❌ Errors ({len(result.errors)}): {', '.join(result.errors[:2])}{'...' if len(result.errors) > 2 else ''}")
        
        total_time = time.time() - self.benchmark_start_time
        summary = self._generate_summary(total_time)
        
        self.file_logger.info(f"Benchmark completed in {total_time:.2f}s")
        print("\n" + "="*60)
        print("📈 BENCHMARK SUMMARY")
        print("="*60)
        self._print_summary(summary)
        
        return summary
    
    def _generate_summary(self, total_time: float) -> BenchmarkSummary:
        """Generate benchmark summary statistics"""
        
        if not self.results:
            return BenchmarkSummary(0, 0.0, 0.0, {}, {}, 1.0, total_time)
        
        scores = [r.verification_score for r in self.results]
        completion_rates = [r.instruction_completion_rate for r in self.results]
        
        # Group by complexity
        scores_by_complexity = {}
        completion_by_complexity = {}
        
        for complexity in range(1, self.config.max_complexity + 1):
            complexity_results = [r for r in self.results if r.complexity_level == complexity]
            if complexity_results:
                scores_by_complexity[complexity] = statistics.mean(
                    [r.verification_score for r in complexity_results]
                )
                completion_by_complexity[complexity] = statistics.mean(
                    [r.instruction_completion_rate for r in complexity_results]
                )
        
        error_count = sum(1 for r in self.results if r.errors)
        
        # Calculate average attempt counts
        avg_content_attempts = statistics.mean([r.content_generation_attempts for r in self.results])
        avg_instruction_attempts = statistics.mean([r.instruction_generation_attempts for r in self.results])
        avg_transformation_attempts = statistics.mean([r.transformation_attempts for r in self.results])
        avg_verification_attempts = statistics.mean([r.verification_attempts for r in self.results])
        
        return BenchmarkSummary(
            total_trials=len(self.results),
            average_score=statistics.mean(scores),
            average_completion_rate=statistics.mean(completion_rates),
            scores_by_complexity=scores_by_complexity,
            completion_rates_by_complexity=completion_by_complexity,
            error_rate=error_count / len(self.results),
            total_time=total_time,
            average_content_attempts=avg_content_attempts,
            average_instruction_attempts=avg_instruction_attempts,
            average_transformation_attempts=avg_transformation_attempts,
            average_verification_attempts=avg_verification_attempts
        )
    
    def _print_summary(self, summary: BenchmarkSummary):
        """Print formatted summary"""
        
        print(f"Total Trials: {summary.total_trials}")
        print(f"Average Quality Score: {summary.average_score:.2f}/10")
        print(f"Average Completion Rate: {summary.average_completion_rate:.1%}")
        print(f"Error Rate: {summary.error_rate:.1%}")
        print(f"Total Time: {summary.total_time:.1f}s")
        print(f"Average Time per Trial: {summary.total_time/summary.total_trials:.1f}s")
        
        print(f"\n📊 Scores by Complexity:")
        for complexity, score in summary.scores_by_complexity.items():
            completion = summary.completion_rates_by_complexity[complexity]
            print(f"  Level {complexity}: {score:.2f}/10 quality, {completion:.1%} completion")
        
        print(f"\n🎯 Performance Trends:")
        complexities = list(summary.scores_by_complexity.keys())
        if len(complexities) > 1:
            score_trend = summary.scores_by_complexity[complexities[-1]] - summary.scores_by_complexity[complexities[0]]
            completion_trend = summary.completion_rates_by_complexity[complexities[-1]] - summary.completion_rates_by_complexity[complexities[0]]
            
            print(f"  Quality Score Change: {score_trend:+.2f} (Level 1 → {complexities[-1]})")
            print(f"  Completion Rate Change: {completion_trend:+.1%} (Level 1 → {complexities[-1]})")
        
        # Show retry statistics if any retries occurred
        if (summary.average_content_attempts > 1.1 or summary.average_instruction_attempts > 1.1 or 
            summary.average_transformation_attempts > 1.1 or summary.average_verification_attempts > 1.1):
            print(f"\n🔄 Retry Statistics:")
            print(f"  Content Generation: {summary.average_content_attempts:.1f} avg attempts")
            print(f"  Instruction Generation: {summary.average_instruction_attempts:.1f} avg attempts")
            print(f"  Transformation: {summary.average_transformation_attempts:.1f} avg attempts")
            print(f"  Verification: {summary.average_verification_attempts:.1f} avg attempts")
    
    def save_detailed_results(self, filename: str = None):
        """Save detailed results to JSON file"""
        
        # Create results directory if it doesn't exist
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        # Ensure filename is in the results directory
        if not filename.startswith(results_dir + os.sep) and not os.path.isabs(filename):
            filename = os.path.join(results_dir, filename)
        
        data = {
            "config": sanitize_config_for_saving(self.config),
            "results": [asdict(result) for result in self.results],
            "summary": asdict(self._generate_summary(0)),
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_trials": len(self.results),
                "log_file": self.file_logger.handlers[0].baseFilename if self.file_logger.handlers else None,
                "llm_configuration": self.llm.get_llm_info()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.file_logger.info(f"Results saved to {filename}")
        print(f"\n💾 Detailed results saved to {filename}")