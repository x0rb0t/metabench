"""
Content and instruction generation classes.
"""

import random
import logging
import time
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.prompts import ChatPromptTemplate
from .llm import BenchmarkLLM
from .parsers import ThinkTagSkippingParser


class EnhancedInstructionGenerator:
    """Generates diverse transformation instructions with expanded variety and retry logic"""
    
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

    def generate_instruction(self, complexity_level: int, content_type: str, max_retries: int = 3) -> Tuple[Dict[str, Any], int]:
        """Generate a diverse transformation instruction with retry logic"""
        
        attempts = 0
        last_exception = None
        
        for attempt in range(max_retries):
            attempts += 1
            try:
                complexity_level = max(1, min(5, complexity_level))
                target = random.choice(self.targets)
                
                self.logger.debug(f"Generating instruction - Attempt {attempt + 1}, Complexity: {complexity_level}, Type: {content_type}")
                self.logger.debug(f"Selected target format: {target}")
                
                if attempt == 0:
                    print(f"    🎯 Target format: {target}")
                else:
                    print(f"    🔄 Retry {attempt}: Generating instruction...")
                
                instruction_parts = {
                    "task": f"Transform {content_type} into {target}",
                    "basic_rules": random.sample(self.formatting_rules, random.randint(2, 5)),
                    "conditional_operations": [],
                    "advanced_processing": [],
                    "verification_points": []
                }
                
                if attempt == 0:
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
                    if attempt == 0:
                        print(f"    🔀 Conditional operations: {len(instruction_parts['conditional_operations'])} added")
                
                # Add advanced processing for higher complexity
                if complexity_level >= 3:
                    num_modifiers = min(complexity_level - 1, len(self.complexity_modifiers))
                    instruction_parts["advanced_processing"] = random.sample(
                        self.complexity_modifiers, random.randint(1, num_modifiers)
                    )
                    if attempt == 0:
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
                if attempt == 0:
                    print(f"    ✅ Instruction generated: {len(verification_points)} verification points")
                else:
                    print(f"    ✅ Instruction generated successfully on attempt {attempt + 1}")
                
                return instruction_parts, attempts
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Instruction generation attempt {attempt + 1} failed: {e}")
                if attempt == 0:
                    print(f"    ⚠️  Instruction generation failed: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.info(f"Retrying instruction generation in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        # All retries exhausted
        self.logger.error(f"All {max_retries} instruction generation attempts failed. Last error: {last_exception}")
        print(f"    ❌ All instruction generation attempts failed")
        raise last_exception


class EnhancedContentGenerator:
    """Generates creative and varied content with random enhancements and retry logic"""
    
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
    
    def generate_content(self, content_type: str, max_retries: int = 3) -> Tuple[str, List[str], int]:
        """Generate content with random creative variations and retry logic"""
        
        attempts = 0
        last_exception = None
        
        for attempt in range(max_retries):
            attempts += 1
            try:
                # Select random number of variations (1-3)
                num_variations = random.randint(1, 3)
                selected_variations = random.sample(self.creative_variations, num_variations)
                
                self.logger.debug(f"Generating {content_type} content - Attempt {attempt + 1} with {num_variations} variations")
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
                
                if attempt == 0:
                    print(f"    🎨 Creative variations: {', '.join(selected_variations)}")
                    print(f"    📝 Enhanced prompt: {enhanced_prompt[:120]}{'...' if len(enhanced_prompt) > 120 else ''}")
                else:
                    print(f"    🔄 Retry {attempt}: Generating content...")
                
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", 
                     "Generate ONLY the requested content. Be creative and follow the variation instructions. "
                     "Output raw content directly without explanations or additional formatting."),
                    ("user", "{prompt}")
                ])
                
                if attempt == 0:
                    print(f"    🤖 Generating enhanced {content_type} content...")
                
                # Use the retry-wrapped LLM call directly
                prompt_input = prompt_template.format_messages(prompt=enhanced_prompt)
                llm_response = self.llm.call_creative_llm(prompt_input)
                response = ThinkTagSkippingParser().parse(llm_response.content)
                
                # Validate response
                if not response or len(response.strip()) < 10:
                    raise ValueError("Generated content is too short or empty")
                
                self.logger.info(f"Generated {len(response)} chars of {content_type} content with variations: {selected_variations}")
                if attempt == 0:
                    print(f"    ✅ Generated {len(response)} chars with {num_variations} creative variations")
                else:
                    print(f"    ✅ Content generated successfully on attempt {attempt + 1}")
                
                return response, selected_variations, attempts
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Content generation attempt {attempt + 1} failed: {e}")
                if attempt == 0:
                    print(f"    ⚠️  Content generation failed: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.info(f"Retrying content generation in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        # All retries exhausted
        self.logger.error(f"All {max_retries} content generation attempts failed. Last error: {last_exception}")
        print(f"    ❌ All content generation attempts failed")
        raise last_exception