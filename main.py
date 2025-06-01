#!/usr/bin/env python3
"""
Self-Evaluating Transformation Benchmark

Final enhanced version with improved logging, progress tracking, and robust error handling.
"""

import json
import random
import statistics
import time
import argparse
import sys
import logging
import os
import getpass
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, BaseOutputParser
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

###############################################################################
# Environment Variable Resolution
###############################################################################

def resolve_env_var(value: str) -> str:
    """
    Resolve environment variable if value starts with 'env:'.
    
    Args:
        value: String that may be in format 'env:VARIABLE_NAME' or regular value
        
    Returns:
        Resolved environment variable value or original value
        
    Raises:
        ValueError: If environment variable is not found
    """
    if value.startswith('env:'):
        env_var_name = value[4:]  # Remove 'env:' prefix
        env_value = os.getenv(env_var_name)
        if env_value is None:
            raise ValueError(f"Environment variable '{env_var_name}' is not set")
        return env_value
    return value

###############################################################################
# Logging Setup
###############################################################################

def setup_logging(log_file: str = None) -> Tuple[logging.Logger, logging.Logger]:
    """Setup dual logging configuration: file-only logger and console-only logger"""
    
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Create unique log filename if none provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"benchmark_log_{timestamp}.log")
    
    # 1. Create FILE-ONLY logger for detailed logging
    file_logger = logging.getLogger('transformation_benchmark_file')
    file_logger.setLevel(logging.DEBUG)
    file_logger.handlers.clear()
    
    # File handler with detailed formatting
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    file_logger.addHandler(file_handler)
    file_logger.propagate = False
    
    # 2. Create CONSOLE-ONLY logger for emoji output
    console_logger = logging.getLogger('transformation_benchmark_console')
    console_logger.setLevel(logging.INFO)
    console_logger.handlers.clear()
    
    # Console handler with minimal formatting (just the message)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    console_logger.addHandler(console_handler)
    console_logger.propagate = False
    
    return file_logger, console_logger

###############################################################################
# Custom Output Parser
###############################################################################

class ThinkTagSkippingParser(BaseOutputParser[str]):
    """Parses out <think> tags and returns the remaining text."""

    def parse(self, text: str) -> str:
        text_without_think_tags = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text_without_think_tags.strip()

    @property
    def _type(self) -> str:
        return "think_tag_skipping_parser"

###############################################################################
# Configuration and Data Classes
###############################################################################

def sanitize_config_for_logging(config: "BenchmarkConfig") -> Dict[str, Any]:
    """Create a sanitized version of config for logging without sensitive data"""
    config_dict = asdict(config)
    # Replace sensitive fields with masked values
    sensitive_fields = ['api_key']
    for field in sensitive_fields:
        if field in config_dict and config_dict[field]:
            # Show only first 4 and last 4 characters if longer than 8, otherwise mask completely
            value = config_dict[field]
            if len(value) > 8:
                config_dict[field] = f"{value[:4]}...{value[-4:]}"
            else:
                config_dict[field] = "***masked***"
    return config_dict

def sanitize_config_for_saving(config: "BenchmarkConfig") -> Dict[str, Any]:
    """Create a sanitized version of config for saving to files without sensitive data"""
    config_dict = asdict(config)
    # Remove sensitive fields entirely from saved data
    sensitive_fields = ['api_key']
    for field in sensitive_fields:
        if field in config_dict:
            config_dict[field] = "***removed_for_security***"
    return config_dict

def validate_config(config: "BenchmarkConfig") -> List[str]:
    """Validate configuration and return list of issues"""
    issues = []
    
    if config.max_complexity < 1 or config.max_complexity > 5:
        issues.append("max_complexity must be between 1 and 5")
    
    if config.trials_per_complexity < 1:
        issues.append("trials_per_complexity must be at least 1")
    
    if not config.content_types:
        issues.append("content_types cannot be empty")
    
    if config.temperature < 0.0 or config.temperature > 2.0:
        issues.append("temperature must be between 0.0 and 2.0")
    
    # Validate content types
    valid_content_types = {"code", "text", "data", "configuration", "documentation"}
    invalid_types = set(config.content_types) - valid_content_types
    if invalid_types:
        issues.append(f"Invalid content types: {', '.join(invalid_types)}. Valid types: {', '.join(valid_content_types)}")
    
    return issues

@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark run"""
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "not-needed"
    temperature: float = 0.3
    max_complexity: int = 5
    trials_per_complexity: int = 3
    content_types: List[str] = None
    topic: Optional[str] = None
    model_name: str = ""
    creative_model: str = ""
    structured_model: str = ""
    transform_model: str = ""
    log_file: str = None
    
    def __post_init__(self):
        if self.content_types is None:
            self.content_types = ["code", "text", "data", "configuration", "documentation"]

@dataclass
class BenchmarkResult:
    """Results from a single benchmark trial"""
    complexity_level: int
    content_type: str
    generated_content: str
    transformation_instruction: str
    transformed_content: str
    verification_score: float
    instruction_completion_rate: float
    execution_time: float
    errors: List[str]
    content_variations: List[str]
    specific_scores: Dict[str, float]

@dataclass
class BenchmarkSummary:
    """Summary of all benchmark results"""
    total_trials: int
    average_score: float
    average_completion_rate: float
    scores_by_complexity: Dict[int, float]
    completion_rates_by_complexity: Dict[int, float]
    error_rate: float
    total_time: float

###############################################################################
# LLM Setup
###############################################################################

class BenchmarkLLM:
    """Wrapper for LLM with different temperature settings for different tasks"""
    
    def __init__(self, config: BenchmarkConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        self.logger.info(f"Initializing LLMs with base URL: {config.base_url}")
        self.logger.info(f"Model configuration: {config.model_name or 'Local/Default'}")
        
        self.creative_llm = ChatOpenAI(
            base_url=config.base_url,
            temperature=0.8,
            api_key=config.api_key,
            model=config.creative_model or config.model_name
        )
        
        self.structured_llm = ChatOpenAI(
            base_url=config.base_url,
            temperature=0.2,
            api_key=config.api_key,
            model=config.structured_model or config.model_name
        )
        
        self.transform_llm = ChatOpenAI(
            base_url=config.base_url,
            temperature=0.1,
            api_key=config.api_key,
            model=config.transform_model or config.model_name
        )

###############################################################################
# Enhanced Instruction Generator
###############################################################################

class EnhancedInstructionGenerator:
    """Generates diverse transformation instructions with expanded variety"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
        # Expanded targets with more diversity
        self.targets = [
            "nested JSON with metadata and timestamps",
            "XML structure with attributes and namespaces", 
            "YAML configuration with inline comments",
            "custom markup with embedded processing directives",
            "hierarchical data tree with parent-child references",
            "serialized object notation with type annotations",
            "structured annotation format with validation schemas",
            "multi-layered dictionary with computed fields",
            "tagged element system with inheritance",
            "hybrid format combining JSON and XML markup",
            "compressed binary-like text representation",
            "graph structure with nodes and edges",
            "relational table format with foreign keys",
            "stream-based event log format",
            "configuration template with variables",
            "documentation format with cross-references",
            "API specification with examples",
            "database schema with constraints",
            "message queue format with routing",
            "indexed search structure with rankings"
        ]
        
        # Enhanced conditions with more variety
        self.conditions = [
            "if the input contains specific keywords or phrases",
            "when encountering special characters or symbols", 
            "if line count exceeds a defined threshold",
            "when finding nested or hierarchical structures",
            "if certain regex patterns are detected",
            "when encountering empty or null sections",
            "if metadata or annotations are present",
            "when finding repeated or duplicate elements",
            "if numeric values exceed certain ranges",
            "when detecting specific file extensions or types",
            "if content contains foreign language text",
            "when finding URLs, emails, or contact information",
            "if timestamps or dates are present",
            "when encountering code blocks or syntax",
            "if content exceeds character limits",
            "when finding mathematical expressions",
            "if content contains media references",
            "when detecting security-sensitive information"
        ]
        
        # Expanded actions with more creativity
        self.actions = [
            "wrap in container object with generated ID",
            "add comprehensive timestamp and version metadata",
            "create indexed references with backlinks", 
            "generate globally unique identifiers",
            "embed processing instructions and directives",
            "create bidirectional cross-reference links",
            "add cryptographic validation markers",
            "insert structural annotations with types",
            "generate computed hash signatures",
            "create hierarchical access control lists",
            "add compression and encoding markers",
            "insert performance optimization hints",
            "create audit trail entries",
            "add internationalization markers",
            "generate dependency mappings",
            "insert caching directives",
            "create rollback checkpoints",
            "add monitoring and alerting hooks"
        ]
        
        # Enhanced formatting rules
        self.formatting_rules = [
            "indent nested elements by exactly 2 spaces",
            "use strict underscore_case for all identifiers",
            "prefix all generated IDs with 'item_' followed by UUID",
            "wrap all text content in CDATA sections",
            "escape special characters using backslash notation",
            "add sequential line numbers to each element",
            "use strict camelCase for all attribute names",
            "maintain original whitespace and preserve formatting",
            "add closing comments for all opening tags",
            "use consistent quote styles throughout",
            "sort all attributes alphabetically",
            "normalize all URLs to absolute paths",
            "convert all timestamps to ISO 8601 format",
            "ensure all text is UTF-8 encoded",
            "add schema validation attributes",
            "use semantic element naming conventions"
        ]
        
        # Enhanced complexity modifiers
        self.complexity_modifiers = [
            "Additionally, create a separate comprehensive index file",
            "Also generate a complete reverse mapping structure", 
            "Include detailed validation rules and schemas in output",
            "Create multiple output format variants simultaneously",
            "Add compression metadata and optimization hints",
            "Generate complete schema definitions with examples",
            "Include detailed processing history and audit logs",
            "Create fully bidirectional reference system with integrity checks",
            "Add performance benchmarks and optimization suggestions",
            "Generate internationalization support structures",
            "Create backup and recovery mechanisms",
            "Add security scanning and validation layers",
            "Generate API documentation and usage examples",
            "Create migration and upgrade pathways",
            "Add monitoring and analytics collection points"
        ]

    def generate_instruction(self, complexity_level: int, content_type: str) -> Dict[str, Any]:
        """Generate a diverse transformation instruction"""
        
        complexity_level = max(1, min(5, complexity_level))
        target = random.choice(self.targets)
        
        self.logger.debug(f"Generating instruction - Complexity: {complexity_level}, Type: {content_type}")
        self.logger.debug(f"Selected target format: {target}")
        
        print(f"    🎯 Target format: {target}")
        
        instruction_parts = {
            "task": f"Transform {content_type} into {target}",
            "basic_rules": random.sample(self.formatting_rules, random.randint(2, 5)),
            "conditional_operations": [],
            "advanced_processing": [],
            "verification_points": []
        }
        
        print(f"    📋 Basic rules: {len(instruction_parts['basic_rules'])} selected")
        
        # Add conditional operations based on complexity
        if complexity_level >= 2:
            num_conditions = min(complexity_level + 1, len(self.conditions))
            for _ in range(random.randint(1, num_conditions)):
                condition = random.choice(self.conditions)
                action = random.choice(self.actions)
                instruction_parts["conditional_operations"].append({
                    "condition": condition,
                    "action": action
                })
            print(f"    🔀 Conditional operations: {len(instruction_parts['conditional_operations'])} added")
        
        # Add advanced processing for higher complexity
        if complexity_level >= 3:
            num_modifiers = min(complexity_level - 1, len(self.complexity_modifiers))
            instruction_parts["advanced_processing"] = random.sample(
                self.complexity_modifiers, random.randint(1, num_modifiers)
            )
            print(f"    ⚡ Advanced processing: {len(instruction_parts['advanced_processing'])} modifiers")
        
        # Create comprehensive verification points
        verification_points = [
            f"Ensure all original {content_type} elements are completely preserved",
            f"Validate that {target} structure is well-formed and compliant",
            "Verify that all conditional rules were properly applied"
        ]
        
        if complexity_level >= 3:
            verification_points.extend([
                "Verify no data loss or corruption occurred during transformation",
                "Confirm all generated IDs are unique and properly formatted"
            ])
        if complexity_level >= 4:
            verification_points.extend([
                "Verify all cross-references are bidirectional and consistent",
                "Confirm metadata consistency across all outputs and variants",
                "Validate performance characteristics meet requirements"
            ])
        if complexity_level >= 5:
            verification_points.extend([
                "Verify security and validation measures are properly implemented",
                "Confirm scalability and optimization features are functional"
            ])
        
        instruction_parts["verification_points"] = verification_points
        
        self.logger.debug(f"Generated instruction with {len(verification_points)} verification points")
        print(f"    ✅ Instruction generated: {len(verification_points)} verification points")
        
        return instruction_parts

###############################################################################
# Enhanced Content Generator
###############################################################################

class EnhancedContentGenerator:
    """Generates creative and varied content with random enhancements"""
    
    def __init__(self, llm: BenchmarkLLM, topic: Optional[str], logger: logging.Logger):
        self.llm = llm
        self.topic = topic
        self.logger = logger
        
        # Creative variations to randomly apply
        self.creative_variations = [
            "include some l33t sp34k or alternative character representations",
            "add special characters and unicode symbols (★, ◆, ♦, ▲, etc.)",
            "include mixed case variations and unusual formatting",
            "add emoji and modern text symbols 🚀 💻 📊",
            "include international characters and accented letters",
            "add intentional typos and informal language",
            "include markdown-style formatting with **bold** and *italic*",
            "add timestamp patterns and version numbers",
            "include URLs, email addresses, and contact information",
            "add technical jargon and domain-specific terminology",
            "include mathematical symbols and formulas",
            "add quoted strings and escaped characters",
            "include nested structures and hierarchical data",
            "add placeholder text and template variables",
            "include comments and annotations"
        ]
        
        self.base_prompts = {
            "code": [
                "Generate a Python class with 3-5 methods implementing a data structure",
                "Create a JavaScript module with functions for data processing",
                "Write a Python script with classes, functions, and main execution",
                "Generate a code snippet with error handling and logging",
                "Create a class hierarchy with inheritance and polymorphism"
            ],
            "text": [
                "Write a 200-300 word article about a technological topic",
                "Create a blog post discussing current trends in technology",
                "Write a technical explanation with examples and comparisons",
                "Generate a news article with quotes and statistics",
                "Create documentation with step-by-step instructions"
            ],
            "data": [
                "Create a realistic CSV dataset with customer information",
                "Generate a JSON structure with nested objects and arrays",
                "Create a database-like table with various data types",
                "Generate sample API response data with metadata",
                "Create a configuration data structure with settings"
            ],
            "configuration": [
                "Create an INI/TOML configuration file for a web application",
                "Generate a YAML configuration with nested settings",
                "Create a JSON config file with environment variables",
                "Generate system configuration with security settings",
                "Create a deployment configuration with multiple environments"
            ],
            "documentation": [
                "Write technical documentation for an API endpoint",
                "Create user documentation with examples and screenshots",
                "Generate developer documentation with code samples",
                "Write installation and setup instructions",
                "Create troubleshooting guide with common issues"
            ]
        }
    
    def generate_content(self, content_type: str) -> Tuple[str, List[str]]:
        """Generate content with random creative variations"""
        
        # Select random number of variations (1-3)
        num_variations = random.randint(1, 3)
        selected_variations = random.sample(self.creative_variations, num_variations)
        
        self.logger.debug(f"Generating {content_type} content with {num_variations} variations")
        self.logger.debug(f"Selected variations: {selected_variations}")
        
        # Build enhanced prompt
        base_prompt = random.choice(self.base_prompts.get(content_type, [
            f"Generate a {content_type} example with realistic structure and content"
        ]))
        
        if self.topic:
            base_prompt = base_prompt.replace("technological", self.topic).replace("technology", self.topic)
            if content_type != "text":
                base_prompt += f" related to {self.topic}"
        
        # Add creative variations
        variations_text = " ADDITIONALLY: " + "; ".join(selected_variations) + "."
        enhanced_prompt = base_prompt + variations_text
        
        print(f"    🎨 Creative variations: {', '.join(selected_variations)}")
        print(f"    📝 Enhanced prompt: {enhanced_prompt[:120]}{'...' if len(enhanced_prompt) > 120 else ''}")
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", 
             "Generate ONLY the requested content. Be creative and follow the variation instructions. "
             "Output raw content directly without explanations or additional formatting."),
            ("user", "{prompt}")
        ])
        
        chain = prompt_template | self.llm.creative_llm | ThinkTagSkippingParser()
        
        print(f"    🤖 Generating enhanced {content_type} content...")
        response = chain.invoke({"prompt": enhanced_prompt})
        
        self.logger.info(f"Generated {len(response)} chars of {content_type} content with variations: {selected_variations}")
        print(f"    ✅ Generated {len(response)} chars with {num_variations} creative variations")
        
        return response, selected_variations

###############################################################################
# Transformation Engine
###############################################################################

class TransformationEngine:
    """Applies transformation instructions to content"""
    
    def __init__(self, llm: BenchmarkLLM, logger: logging.Logger):
        self.llm = llm
        self.logger = logger
    
    def apply_transformation(self, content: str, instruction: Dict[str, Any]) -> str:
        """Apply transformation instruction to content"""
        
        instruction_text = self._format_instruction(instruction)
        
        self.logger.debug(f"Applying transformation: {instruction['task']}")
        self.logger.debug(f"Instruction length: {len(instruction_text)} chars")
        
        print(f"    📝 Transformation instruction formatted ({len(instruction_text)} chars)")
        print(f"    🎯 Task: {instruction['task']}")
        print(f"    📋 Rules: {len(instruction['basic_rules'])} basic, {len(instruction.get('conditional_operations', []))} conditional")
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a precise transformation engine. Apply the transformation instructions exactly. "
             "Follow ALL rules and requirements. Output only the transformed content without explanations."),
            ("user", 
             "Original content:\n{content}\n\n"
             "Transformation instructions:\n{instruction}\n\n"
             "Apply the transformation and output the result:")
        ])
        
        chain = prompt_template | self.llm.transform_llm | ThinkTagSkippingParser()
        
        print(f"    ⚙️  Invoking transformation chain...")
        response = chain.invoke({
            "content": content,
            "instruction": instruction_text
        })
        
        self.logger.info(f"Transformation completed: {len(content)} → {len(response)} chars")
        print(f"    ✅ Transformation completed ({len(response)} chars output)")
        
        return response
    
    def _format_instruction(self, instruction: Dict[str, Any]) -> str:
        """Format instruction dictionary as human-readable text"""
        
        text = f"TASK: {instruction['task']}\n\n"
        
        text += "BASIC RULES:\n"
        for rule in instruction['basic_rules']:
            text += f"- {rule.capitalize()}\n"
        
        if instruction['conditional_operations']:
            text += "\nCONDITIONAL OPERATIONS:\n"
            for op in instruction['conditional_operations']:
                text += f"- {op['condition'].capitalize()}, then {op['action']}\n"
        
        if instruction['advanced_processing']:
            text += "\nADVANCED PROCESSING:\n"
            for proc in instruction['advanced_processing']:
                text += f"- {proc}\n"
        
        text += "\nVERIFICATION REQUIREMENTS:\n"
        for point in instruction['verification_points']:
            text += f"- {point}\n"
        
        return text

###############################################################################
# Enhanced Verification Engine
###############################################################################

class EnhancedVerificationEngine:
    """Verifies and scores transformation results with improved logic"""
    
    def __init__(self, llm: BenchmarkLLM, logger: logging.Logger):
        self.llm = llm
        self.logger = logger
    
    def verify_transformation(self, 
                            original_content: str,
                            instruction: Dict[str, Any],
                            transformed_content: str) -> Tuple[float, float, Dict[str, float]]:
        """Verify transformation and return (quality_score, completion_rate, specific_scores)"""
        
        instruction_text = self._format_verification_criteria(instruction)
        requirements_list = self._extract_requirements_list(instruction)
        applicable_categories = self._get_applicable_categories(instruction)
        
        self.logger.debug(f"Verifying transformation with {len(requirements_list)} requirements")
        self.logger.debug(f"Applicable categories: {applicable_categories}")
        
        print(f"    🔍 Verification criteria: {len(instruction_text)} chars")
        print(f"    📊 Comparing {len(original_content)} → {len(transformed_content)} chars")
        print(f"    📋 Requirements to verify: {len(requirements_list)} items")
        print(f"    🎯 Applicable categories: {len(applicable_categories)}")
        
        verification_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a strict verification engine. Analyze transformation results objectively. "
             "Score only the applicable categories. Return ONLY valid JSON in the specified format."),
            ("user",
             "Original content:\n{original}\n\n"
             "Transformation instructions:\n{instruction}\n\n" 
             "Transformed result:\n{transformed}\n\n"
             "Requirements to evaluate:\n{requirements}\n\n"
             "Applicable categories: {categories}\n\n"
             "Return JSON with this EXACT structure:\n"
             "{{\n"
             '  "specific_scores": {{\n'
             '    "basic_rules_score": <float 0-10 or null if not applicable>,\n'
             '    "conditional_operations_score": <float 0-10 or null if not applicable>,\n'
             '    "advanced_processing_score": <float 0-10 or null if not applicable>,\n'
             '    "verification_requirements_score": <float 0-10>,\n'
             '    "data_preservation_score": <float 0-10>,\n'
             '    "format_compliance_score": <float 0-10>\n'
             '  }},\n'
             '  "quality_score": <float 0-10>,\n'
             '  "instruction_completion": <float 0-1>,\n'
             '  "feedback": "<detailed explanation>"\n'
             "}}\n\n"
             "Score only applicable categories. Use null for non-applicable categories.")
        ])
        
        chain = verification_prompt | self.llm.structured_llm | ThinkTagSkippingParser() | JsonOutputParser()
        
        try:
            print(f"    ⚙️  Running verification analysis...")
            result = chain.invoke({
                "original": original_content,
                "instruction": instruction_text,
                "transformed": transformed_content,
                "requirements": "\n".join([f"- {req}" for req in requirements_list]),
                "categories": ", ".join(applicable_categories)
            })
            
            # Process specific scores with proper null handling
            specific_scores = self._process_specific_scores(
                result.get('specific_scores', {}), 
                applicable_categories
            )
            
            # Extract and validate overall scores
            quality_score = self._validate_float_score(result.get('quality_score', 0), 0, 10, "quality_score")
            completion_rate = self._validate_float_score(result.get('instruction_completion', 0), 0, 1, "instruction_completion")
            
            # Calculate average only from applicable categories
            applicable_scores = [score for cat, score in specific_scores.items() 
                               if cat in applicable_categories and score is not None]
            avg_specific = sum(applicable_scores) / len(applicable_scores) if applicable_scores else 0
            
            self.logger.info(f"Verification complete: Quality={quality_score:.2f}, Completion={completion_rate:.2%}")
            self.logger.info(f"Specific scores: {specific_scores}")
            
            print(f"    ✅ Verification complete: Quality={quality_score:.2f}/10, Completion={completion_rate:.2%}")
            
            # Log each specific score with applicability
            for category, score in specific_scores.items():
                if category in applicable_categories:
                    print(f"      🎯 {category}: {score:.2f}/10")
                else:
                    print(f"      ⚫ {category}: N/A (not applicable)")
            
            feedback = result.get('feedback', 'No feedback provided')
            print(f"    💬 Feedback: {feedback[:150]}{'...' if len(feedback) > 150 else ''}")
            print(f"    📈 Average applicable score: {avg_specific:.2f}/10")
            
            return quality_score, completion_rate, specific_scores
            
        except (ConnectionError, TimeoutError) as e:
            self.logger.error(f"Network error during verification: {e}")
            print(f"    🌐 Network error during verification: {e}")
            return 0.0, 0.0, {}
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error in verification: {e}")
            print(f"    📄 JSON parsing error in verification: {e}")
            return 0.0, 0.0, {}
        except Exception as e:
            self.logger.error(f"Verification failed: {e}", exc_info=True)
            print(f"    ❌ Verification error: {e}")
            return 0.0, 0.0, {}
    
    def _get_applicable_categories(self, instruction: Dict[str, Any]) -> List[str]:
        """Determine which scoring categories are applicable for this instruction"""
        applicable = ['verification_requirements_score', 'data_preservation_score', 'format_compliance_score']
        
        if instruction.get('basic_rules'):
            applicable.append('basic_rules_score')
        
        if instruction.get('conditional_operations'):
            applicable.append('conditional_operations_score')
        
        if instruction.get('advanced_processing'):
            applicable.append('advanced_processing_score')
        
        return applicable
    
    def _process_specific_scores(self, specific_scores: Dict[str, Any], applicable_categories: List[str]) -> Dict[str, float]:
        """Process specific scores, handling null values for non-applicable categories"""
        processed_scores = {}
        
        all_categories = [
            'basic_rules_score', 'conditional_operations_score', 'advanced_processing_score',
            'verification_requirements_score', 'data_preservation_score', 'format_compliance_score'
        ]
        
        for category in all_categories:
            score_value = specific_scores.get(category)
            
            if category in applicable_categories:
                # This category should be scored
                if score_value is None:
                    self.logger.warning(f"Applicable category {category} was null, using 0.0")
                    processed_scores[category] = 0.0
                else:
                    processed_scores[category] = self._validate_float_score(score_value, 0, 10, category)
            else:
                # This category is not applicable
                processed_scores[category] = None
        
        return processed_scores
    
    def _extract_requirements_list(self, instruction: Dict[str, Any]) -> List[str]:
        """Extract a flat list of all requirements from instruction"""
        requirements = []
        
        requirements.extend(instruction.get('basic_rules', []))
        
        for op in instruction.get('conditional_operations', []):
            requirements.append(f"{op.get('condition', '')} → {op.get('action', '')}")
        
        requirements.extend(instruction.get('advanced_processing', []))
        requirements.extend(instruction.get('verification_points', []))
        
        return requirements
    
    def _validate_float_score(self, value: Any, min_val: float, max_val: float, field_name: str) -> float:
        """Validate and clamp a score value to be a float within range"""
        try:
            float_value = float(value)
            clamped_value = max(min_val, min(max_val, float_value))
            
            if float_value != clamped_value:
                self.logger.warning(f"{field_name} clamped from {float_value} to {clamped_value}")
            
            return clamped_value
            
        except (ValueError, TypeError) as e:
            self.logger.error(f"Invalid {field_name} value '{value}': {e}")
            return 0.0
    
    def _format_verification_criteria(self, instruction: Dict[str, Any]) -> str:
        """Format instruction as verification criteria"""
        
        criteria = f"Task: {instruction['task']}\n\n"
        
        if instruction.get('basic_rules'):
            criteria += "BASIC RULES TO VERIFY:\n"
            for rule in instruction['basic_rules']:
                criteria += f"- {rule.capitalize()}\n"
        
        if instruction.get('conditional_operations'):
            criteria += "\nCONDITIONAL OPERATIONS TO VERIFY:\n"
            for op in instruction['conditional_operations']:
                criteria += f"- {op['condition'].capitalize()}, then {op['action']}\n"
        
        if instruction.get('advanced_processing'):
            criteria += "\nADVANCED PROCESSING TO VERIFY:\n"
            for proc in instruction['advanced_processing']:
                criteria += f"- {proc}\n"
        
        criteria += "\nVERIFICATION REQUIREMENTS:\n"
        for point in instruction['verification_points']:
            criteria += f"- {point}\n"
        
        return criteria

###############################################################################
# Main Benchmark Runner
###############################################################################

class TransformationBenchmark:
    """Main benchmark runner with enhanced logging, progress tracking, and scoring"""
    
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
        """Run a single benchmark trial with comprehensive logging and progress tracking"""
        
        start_time = time.time()
        errors = []
        
        # Enhanced trial start logging for file only
        self.file_logger.info("=" * 50)
        self.file_logger.info(f"TRIAL START: Complexity Level {complexity_level}, Content Type: {content_type}")
        self.file_logger.info(f"Trial {self.current_trial + 1}/{self.total_trials}")
        self.file_logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.file_logger.info("=" * 50)
        
        print(f"    🚀 Starting trial: Complexity {complexity_level}, Type {content_type}")
        
        try:
            # Phase 1: Generate content
            print(f"  📝 Phase 1/4: Generating {content_type} content...")
            
            self.file_logger.info("-" * 30)
            self.file_logger.info(f"PHASE 1 START: Content Generation ({content_type})")
            self.file_logger.info("-" * 30)
            
            generated_content, content_variations = self.content_generator.generate_content(content_type)
            
            self.file_logger.info(f"Generated content length: {len(generated_content)} characters")
            self.file_logger.info(f"Content variations applied: {content_variations}")
            self._log_content_safely(self.file_logger, generated_content, "GENERATED CONTENT")
            self.file_logger.info(f"PHASE 1 COMPLETE: Content Generation")
            self.file_logger.info("-" * 30)
            
            # Phase 2: Generate transformation instruction
            print(f"  🎯 Phase 2/4: Creating complexity {complexity_level} transformation...")
            
            self.file_logger.info("-" * 30)
            self.file_logger.info(f"PHASE 2 START: Instruction Generation (Complexity {complexity_level})")
            self.file_logger.info("-" * 30)
            
            instruction = self.instruction_generator.generate_instruction(complexity_level, content_type)
            
            self.file_logger.info("INSTRUCTION START:")
            self.file_logger.info(json.dumps(instruction, indent=2))
            self.file_logger.info("INSTRUCTION END")
            self.file_logger.info(f"PHASE 2 COMPLETE: Instruction Generation")
            self.file_logger.info("-" * 30)
            
            # Phase 3: Apply transformation
            print(f"  ⚙️  Phase 3/4: Applying transformation...")
            
            self.file_logger.info("-" * 30)
            self.file_logger.info("PHASE 3 START: Transformation Application")
            self.file_logger.info("-" * 30)
            self.file_logger.info(f"Input content length: {len(generated_content)} characters")
            
            transformed_content = self.transformation_engine.apply_transformation(
                generated_content, instruction
            )
            
            self.file_logger.info(f"Output content length: {len(transformed_content)} characters")
            self._log_content_safely(self.file_logger, transformed_content, "TRANSFORMED CONTENT")
            self.file_logger.info("PHASE 3 COMPLETE: Transformation Application")
            self.file_logger.info("-" * 30)
            
            # Phase 4: Verify results
            print(f"  🔍 Phase 4/4: Verifying results...")
            
            self.file_logger.info("-" * 30)
            self.file_logger.info("PHASE 4 START: Verification")
            self.file_logger.info("-" * 30)
            
            quality_score, completion_rate, specific_scores = self.verification_engine.verify_transformation(
                generated_content, instruction, transformed_content
            )
            
            self.file_logger.info(f"Quality Score: {quality_score:.2f}/10")
            self.file_logger.info(f"Completion Rate: {completion_rate:.2%}")
            self.file_logger.info(f"Specific Scores: {specific_scores}")
            self.file_logger.info("PHASE 4 COMPLETE: Verification")
            self.file_logger.info("-" * 30)
            
        except (ConnectionError, TimeoutError) as e:
            error_msg = f"Network error: {str(e)}"
            self.file_logger.error(f"TRIAL FAILURE: {error_msg}")
            print(f"    🌐 {error_msg}")
            errors.append(error_msg)
            generated_content, content_variations, instruction, transformed_content = "", [], {}, ""
            quality_score, completion_rate, specific_scores = 0.0, 0.0, {}
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing error: {str(e)}"
            self.file_logger.error(f"TRIAL FAILURE: {error_msg}")
            print(f"    📄 {error_msg}")
            errors.append(error_msg)
            generated_content, content_variations, instruction, transformed_content = "", [], {}, ""
            quality_score, completion_rate, specific_scores = 0.0, 0.0, {}
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
        
        execution_time = time.time() - start_time
        
        # Enhanced trial completion logging for file only
        self.file_logger.info("=" * 50)
        self.file_logger.info(f"TRIAL COMPLETE: Execution time {execution_time:.2f}s")
        self.file_logger.info(f"Final Results - Quality: {quality_score:.2f}/10, Completion: {completion_rate:.2%}")
        if errors:
            self.file_logger.info(f"Errors encountered: {len(errors)}")
        self.file_logger.info("=" * 50)
        
        print(f"    ⏱️  Trial completed in {execution_time:.2f}s")
        
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
            specific_scores=specific_scores
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
        print(f"Base URL: {self.config.base_url}")
        print(f"Model: {self.config.model_name or 'Local/Default'}")
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
        
        return BenchmarkSummary(
            total_trials=len(self.results),
            average_score=statistics.mean(scores),
            average_completion_rate=statistics.mean(completion_rates),
            scores_by_complexity=scores_by_complexity,
            completion_rates_by_complexity=completion_by_complexity,
            error_rate=error_count / len(self.results),
            total_time=total_time
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
                "log_file": self.file_logger.handlers[0].baseFilename if self.file_logger.handlers else None
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.file_logger.info(f"Results saved to {filename}")
        print(f"\n💾 Detailed results saved to {filename}")

###############################################################################
# Interactive Mode and CLI
###############################################################################

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