"""
Content and instruction generation classes with significantly enhanced diversity and creativity.
"""

import random
import logging
import time
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.prompts import ChatPromptTemplate
from .llm import BenchmarkLLM
from .parsers import ThinkTagSkippingParser


class EnhancedInstructionGenerator:
    """Generates coherent transformation instructions with exponential complexity scaling and enhanced diversity"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
        # Enhanced transformation families with significantly more diverse options
        self.transformation_families = {
            1: {  # Level 1: Simple Structural
                "targets": [
                    "indented list format with bullet points and special markers",
                    "numbered outline structure with roman numerals and sub-levels",
                    "simple CSV table format with headers and data alignment",
                    "basic JSON object structure with nested properties",
                    "plain text with line breaks and section dividers",
                    "markdown format with headers and emphasis markers",
                    "structured outline with alphanumeric indexing system",
                    "tabular format with column separators and borders",
                    "hierarchical tree structure with branch indicators",
                    "definition list format with terms and descriptions"
                ],
                "basic_rules": [
                    "preserve all original content exactly without modification",
                    "maintain original word order and semantic relationships",
                    "use consistent indentation of exactly 2 spaces per level",
                    "separate distinct items with clear line breaks",
                    "add simple structural markers like bullets or numbers",
                    "ensure proper alignment of all text elements",
                    "include clear section boundaries and delimiters",
                    "maintain consistent capitalization throughout",
                    "apply uniform spacing between all elements",
                    "preserve punctuation and special characters intact"
                ],
                "conditions": [],
                "actions": [],
                "advanced_processing": []
            },
            2: {  # Level 2: Enhanced Structural + Basic Conditions
                "targets": [
                    "structured JSON with comprehensive metadata headers and timestamps",
                    "XML format with proper tag hierarchy and attribute validation",
                    "markdown document with headers, formatting, and cross-references",
                    "YAML configuration with nested sections and inline comments",
                    "formatted table with aligned columns and visual separators",
                    "annotated outline with detailed explanations and examples",
                    "structured database schema with relationships and constraints",
                    "technical specification with parameters and validation rules",
                    "formatted report with sections, subsections, and appendices",
                    "interactive checklist with status indicators and progress tracking"
                ],
                "basic_rules": [
                    "add comprehensive timestamp metadata to each major section",
                    "include globally unique identifiers for every content block",
                    "maintain strict hierarchical structure relationships throughout",
                    "escape all special characters using proper encoding standards",
                    "add detailed content type annotations with schema validation",
                    "implement consistent naming conventions across all elements",
                    "ensure backward compatibility with original data structures",
                    "apply standardized formatting rules for visual consistency",
                    "include version control information and change tracking",
                    "maintain referential integrity between related components"
                ],
                "conditions": [
                    "if content contains more than 5 lines of continuous text",
                    "when encountering numbered items or sequential elements",
                    "if text contains technical keywords or domain-specific terminology",
                    "when finding quoted text, code blocks, or formatted content",
                    "if content includes dates, times, or temporal references",
                    "when discovering URLs, email addresses, or contact information",
                    "if text contains statistical data or numerical measurements",
                    "when encountering foreign language text or transliterations",
                    "if content includes mathematical expressions or formulas",
                    "when finding references to external sources or citations"
                ],
                "actions": [
                    "wrap in secure container with generated UUID and metadata",
                    "add hierarchical section headers with automatic numbering",
                    "create bidirectional cross-reference links with anchor points",
                    "include comprehensive word count and readability metrics",
                    "generate searchable index with keywords and categories",
                    "add validation checkpoints with error detection mechanisms",
                    "create summary abstracts for lengthy content sections",
                    "implement tagging system with multiple classification levels",
                    "generate table of contents with clickable navigation",
                    "add accessibility features with alternative text descriptions"
                ],
                "advanced_processing": []
            },
            3: {  # Level 3: Cross-Format + Semantic Elements
                "targets": [
                    "immersive narrative story with rich character development and atmospheric scenes",
                    "comprehensive technical documentation with detailed examples and troubleshooting guides",
                    "expressive poetry with sophisticated meter, rhyme schemes, and literary devices",
                    "formal academic paper with rigorous methodology and scholarly citations",
                    "interactive multimedia tutorial with step-by-step progression and checkpoints",
                    "detailed product review with comparative analysis and expert recommendations",
                    "philosophical dialogue with contrasting viewpoints and Socratic questioning",
                    "dramatic screenplay with character arcs and visual storytelling elements",
                    "investigative journalism piece with sources, evidence, and narrative structure",
                    "educational case study with real-world applications and learning objectives"
                ],
                "basic_rules": [
                    "transform writing style completely from source to target genre conventions",
                    "maintain semantic core while revolutionizing presentation methodology",
                    "add genre-appropriate elements like dialogue, citations, or technical specifications",
                    "create compelling narrative flow with clear beginning, development, and resolution",
                    "include extensive contextual background that enriches understanding",
                    "establish authentic voice consistent with target format expectations",
                    "integrate supporting evidence and examples throughout the text",
                    "develop thematic coherence that unifies all content elements",
                    "implement sophisticated organizational structure appropriate to genre",
                    "ensure cultural and contextual appropriateness for intended audience"
                ],
                "conditions": [
                    "when content describes complex processes, procedures, or methodologies",
                    "if content contains detailed technical specifications or requirements",
                    "when encountering quantitative data, statistics, or measurement values",
                    "if content has clear logical sequences or temporal progressions",
                    "when finding abstract concepts, theories, or philosophical ideas",
                    "if content involves human experiences, emotions, or personal narratives",
                    "when encountering conflicting viewpoints or controversial topics",
                    "if content includes scientific research or experimental data",
                    "when finding historical references or chronological information",
                    "if content involves creative or artistic expressions and interpretations"
                ],
                "actions": [
                    "reframe information as compelling characters with motivations and conflicts",
                    "convert abstract data into vivid descriptive examples with sensory details",
                    "transform technical jargon into accessible language with clear explanations",
                    "create sophisticated metaphors and analogies for complex conceptual frameworks",
                    "add profound emotional resonance and experiential context to all elements",
                    "establish dramatic tension and narrative momentum throughout the piece",
                    "develop multiple perspective layers with varying levels of expertise",
                    "integrate historical context and evolutionary development patterns",
                    "create interactive elements that engage readers in active participation",
                    "build thematic connections that reveal deeper meaning and significance"
                ],
                "advanced_processing": [
                    "ensure strict adherence to genre conventions and stylistic expectations",
                    "maintain rigorous factual accuracy while dramatically enhancing presentation quality",
                    "create seamless integration between original content and creative enhancements",
                    "establish consistent tone and voice appropriate for target audience",
                    "implement quality assurance checks for logical coherence and flow"
                ]
            },
            4: {  # Level 4: Complex Multi-Stage Transformations
                "targets": [
                    "sophisticated multi-perspective analysis with contrasting expert viewpoints and synthesis",
                    "dynamic interactive dialogue between multiple specialized personas with distinct expertise",
                    "complex layered narrative with interweaving main story and meaningful subplots",
                    "comprehensive analytical report with data visualizations and strategic recommendations",
                    "profound philosophical examination with rigorous arguments and thoughtful counterarguments",
                    "innovative creative reinterpretation as authentic historical document or speculative future artifact",
                    "immersive simulation scenario with multiple stakeholders and decision points",
                    "scholarly debate format with evidence-based positions and logical rebuttals",
                    "investigative documentary structure with interviews, evidence, and narrative progression",
                    "strategic business case with market analysis, risk assessment, and implementation roadmap"
                ],
                "basic_rules": [
                    "present comprehensive content from multiple distinct professional perspectives",
                    "maintain absolute logical consistency across all viewpoints and arguments",
                    "create seamless transitions between different sections and perspective shifts",
                    "establish meaningful relationships and connections between all content elements",
                    "build sophisticated narrative or analytical architecture with clear progression",
                    "ensure each perspective adds unique value and authentic expertise",
                    "develop compelling argumentation chains with supporting evidence",
                    "integrate multiple layers of analysis from surface to deep structural levels",
                    "create dynamic tension and intellectual engagement throughout the piece",
                    "establish clear resolution or synthesis that honors all perspectives"
                ],
                "conditions": [
                    "when content can be analyzed from multiple professional or disciplinary viewpoints",
                    "if content involves complex decisions with significant trade-offs and implications",
                    "when content reveals cause-and-effect relationships with multiple variables",
                    "if content relates to human experiences with emotional and social dimensions",
                    "when content involves complex systems with interconnected components",
                    "if content contains controversial elements requiring balanced treatment",
                    "when content has historical precedents or future implications",
                    "if content involves ethical considerations or moral dimensions",
                    "when content requires interdisciplinary knowledge and expertise",
                    "if content has practical applications with real-world consequences"
                ],
                "actions": [
                    "create authentic expert personas with detailed backgrounds and specialized knowledge",
                    "develop comprehensive argumentation chains with evidence and logical progression",
                    "establish dynamic dialogue and intellectual debate between contrasting viewpoints",
                    "create extensive supporting evidence including examples, case studies, and data",
                    "build compelling narrative tension with intellectual and emotional engagement",
                    "integrate multiple analytical layers from technical to philosophical dimensions",
                    "develop sophisticated synthesis that transcends individual perspectives",
                    "create interactive decision points that engage readers in critical thinking",
                    "establish historical context and future implications for all major points",
                    "build toward transformative insights that emerge from the analytical process"
                ],
                "advanced_processing": [
                    "ensure each perspective is thoroughly developed with authentic depth and expertise",
                    "create meaningful intellectual connections and productive contrasts between viewpoints",
                    "maintain sustained reader engagement across multiple complex sections",
                    "build toward genuine synthesis or deeper understanding that justifies the complexity",
                    "implement rigorous fact-checking and logical consistency validation throughout"
                ]
            },
            5: {  # Level 5: Creative Reinterpretation + Meta-Transformation
                "targets": [
                    "recursive meta-commentary where content becomes self-aware and analyzes its own transformation",
                    "genre-blending fusion that seamlessly combines multiple creative forms and traditions",
                    "temporal recontextualization as authentic historical artifact from entirely different era",
                    "consciousness stream of artificial intelligence analyzing and reinterpreting the source material",
                    "parallel universe version where fundamental assumptions about reality are completely different",
                    "collaborative conversation between the original content and its evolving transformation",
                    "philosophical meditation on the nature of information and meaning itself",
                    "time-traveling narrative where content exists across multiple historical periods",
                    "fractal structure where each part contains and reflects the essence of the whole",
                    "quantum superposition format where multiple interpretations exist simultaneously"
                ],
                "basic_rules": [
                    "completely recontextualize while preserving essential informational DNA",
                    "create multiple simultaneous transformation layers with recursive depth",
                    "establish revolutionary perspectives that reveal previously hidden insights",
                    "maintain perfect internal consistency within the new reality framework",
                    "challenge and transcend conventional presentation and communication assumptions",
                    "integrate meta-textual elements that comment on the transformation process itself",
                    "create philosophical depth that questions the nature of information and meaning",
                    "establish temporal or dimensional frameworks that redefine context completely",
                    "build collaborative relationships between different levels of consciousness",
                    "achieve transcendent understanding that encompasses and surpasses all individual perspectives"
                ],
                "conditions": [
                    "when content represents fundamental human knowledge, creativity, or understanding",
                    "if content embodies implicit assumptions about reality, society, or consciousness",
                    "when content has embedded cultural, temporal, or philosophical context",
                    "if content involves communication, expression, or knowledge transfer",
                    "when content embodies particular worldviews, methodologies, or belief systems",
                    "if content can be viewed as artifact of specific time, place, or mindset",
                    "when content contains recursive or self-referential elements",
                    "if content involves questions of meaning, purpose, or significance",
                    "when content has potential for multiple valid interpretations",
                    "if content touches on universal themes or archetypal patterns"
                ],
                "actions": [
                    "create alternate reality frameworks where content exists in completely different contexts",
                    "establish profound dialogue between content and its own transformation process",
                    "develop meta-narrative about the fundamental nature of information and communication",
                    "create recursive loops where content comments on its own evolution and meaning",
                    "establish temporal or dimensional shifts that completely reframe all understanding",
                    "build collaborative creation process between multiple levels of consciousness",
                    "develop philosophical frameworks that question basic assumptions about reality",
                    "create quantum entanglement between different versions of the content",
                    "establish archetypal patterns that connect to universal human experiences",
                    "build toward transcendent synthesis that reveals the unity underlying all perspectives"
                ],
                "advanced_processing": [
                    "maintain philosophical coherence across all transformation layers and dimensions",
                    "create genuine insights through recontextualization that enhance rather than obscure truth",
                    "ensure transformation reveals rather than conceals the essential nature of the content",
                    "establish meaningful dialogue between different reality frameworks and consciousness levels",
                    "build toward transcendent understanding that encompasses all perspectives while transcending limitations",
                    "implement recursive validation where each layer confirms and enriches all others",
                    "create emergent properties that arise from but exceed the sum of individual components"
                ]
            }
        }
        
        # Enhanced content-aware enhancement patterns
        self.content_enhancements = {
            "creative": [
                "include rich sensory details with synesthetic descriptions and atmospheric elements",
                "add profound emotional resonance with character development and psychological depth",
                "create vivid scene setting with immersive environmental and cultural details",
                "establish compelling narrative voice with unique perspective and authentic personality",
                "integrate symbolic and metaphorical language with multiple layers of meaning",
                "develop thematic coherence with universal archetypal patterns and motifs",
                "create dialogue that reveals character while advancing plot and theme",
                "establish temporal and spatial frameworks that enhance narrative immersion"
            ],
            "technical": [
                "include precise technical specifications with implementation details and constraints",
                "add comprehensive error handling scenarios with edge cases and recovery strategies",
                "create detailed implementation examples with best practices and optimization techniques",
                "establish clear architectural relationships with dependencies and interaction patterns",
                "integrate performance metrics with benchmarks and scalability considerations",
                "develop security protocols with threat modeling and mitigation strategies",
                "create comprehensive testing frameworks with validation and verification procedures",
                "establish documentation standards with clear examples and usage guidelines"
            ],
            "analytical": [
                "include rigorous statistical analysis with confidence intervals and significance testing",
                "add logical argumentation chains with evidence evaluation and reasoning validation",
                "create evidence-based conclusions with supporting data and methodological transparency",
                "establish methodological rigor with peer review standards and replication protocols",
                "integrate comparative analysis with control groups and baseline measurements",
                "develop causal inference frameworks with confounding variable identification",
                "create predictive models with uncertainty quantification and sensitivity analysis",
                "establish data visualization standards with clear interpretation guidelines"
            ]
        }

    def generate_instruction(self, complexity_level: int, content_type: str, max_retries: int = 3) -> Tuple[Dict[str, Any], int]:
        """Generate coherent transformation instruction with exponential complexity scaling"""
        
        attempts = 0
        last_exception = None
        
        for attempt in range(max_retries):
            attempts += 1
            try:
                complexity_level = max(1, min(5, complexity_level))
                family = self.transformation_families[complexity_level]
                
                self.logger.debug(f"Generating instruction - Attempt {attempt + 1}, Complexity: {complexity_level}, Type: {content_type}")
                
                # Select coherent target based on content type and complexity
                target = self._select_coherent_target(family["targets"], content_type, complexity_level)
                
                if attempt == 0:
                    print(f"    🎯 Target format: {target}")
                    print(f"    📊 Complexity level {complexity_level} transformation")
                else:
                    print(f"    🔄 Retry {attempt}: Generating instruction...")
                
                instruction_parts = {
                    "task": f"Transform {content_type} into {target}",
                    "basic_rules": self._select_basic_rules(family["basic_rules"], complexity_level),
                    "conditional_operations": self._generate_conditional_operations(family, complexity_level),
                    "advanced_processing": self._select_advanced_processing(family["advanced_processing"], complexity_level),
                    "verification_points": self._generate_verification_points(target, complexity_level, content_type)
                }
                
                # Add content-type specific enhancements
                instruction_parts = self._add_content_enhancements(instruction_parts, content_type, complexity_level)
                
                if attempt == 0:
                    print(f"    📋 Rules: {len(instruction_parts['basic_rules'])} basic, {len(instruction_parts['conditional_operations'])} conditional")
                    print(f"    ⚡ Advanced processing: {len(instruction_parts['advanced_processing'])} elements")
                else:
                    print(f"    ✅ Instruction generated successfully on attempt {attempt + 1}")
                
                self.logger.debug(f"Generated {complexity_level}-level instruction with {len(instruction_parts['verification_points'])} verification points")
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
    
    def _select_coherent_target(self, targets: List[str], content_type: str, complexity_level: int) -> str:
        """Select target that makes sense for the content type"""
        # For creative content types, prefer creative targets
        creative_content = ["poetry", "short_story", "song_lyrics", "screenplay", "philosophical_essay"]
        creative_targets = ["narrative story", "poetry", "dialogue", "philosophical examination", "screenplay"]
        
        if content_type in creative_content and complexity_level >= 3:
            suitable_targets = [t for t in targets if any(ct in t for ct in creative_targets)]
            if suitable_targets:
                return random.choice(suitable_targets)
        
        return random.choice(targets)
    
    def _select_basic_rules(self, available_rules: List[str], complexity_level: int) -> List[str]:
        """Select basic rules with exponential scaling"""
        # Scale: 1-2 rules for level 1, up to 5+ rules for level 5
        min_rules = complexity_level
        max_rules = min(complexity_level + 2, len(available_rules))
        num_rules = random.randint(min_rules, max_rules)
        return random.sample(available_rules, min(num_rules, len(available_rules)))
    
    def _generate_conditional_operations(self, family: Dict, complexity_level: int) -> List[Dict[str, str]]:
        """Generate conditional operations with complexity scaling"""
        if complexity_level < 2:
            return []
        
        conditions = family.get("conditions", [])
        actions = family.get("actions", [])
        
        if not conditions or not actions:
            return []
        
        # Exponential scaling: 0, 1-2, 2-3, 3-4, 4-5 conditions
        num_conditions = min(complexity_level, random.randint(1, complexity_level + 1))
        
        operations = []
        for _ in range(num_conditions):
            operations.append({
                "condition": random.choice(conditions),
                "action": random.choice(actions)
            })
        
        return operations
    
    def _select_advanced_processing(self, available_processing: List[str], complexity_level: int) -> List[str]:
        """Select advanced processing with complexity scaling"""
        if complexity_level < 3 or not available_processing:
            return []
        
        # Scale advanced processing exponentially
        num_processing = min(complexity_level - 2, len(available_processing))
        return random.sample(available_processing, num_processing)
    
    def _add_content_enhancements(self, instruction_parts: Dict, content_type: str, complexity_level: int) -> Dict:
        """Add content-type specific enhancements"""
        if complexity_level < 3:
            return instruction_parts
        
        # Determine content category
        creative_types = ["poetry", "short_story", "song_lyrics", "screenplay", "philosophical_essay", "interview_transcript"]
        technical_types = ["code", "api_documentation", "database_schema", "system_logs", "configuration"]
        analytical_types = ["scientific_paper", "case_study", "product_review", "data"]
        
        enhancement_type = "creative" if content_type in creative_types else \
                          "technical" if content_type in technical_types else \
                          "analytical" if content_type in analytical_types else "creative"
        
        enhancements = self.content_enhancements.get(enhancement_type, [])
        if enhancements and complexity_level >= 4:
            instruction_parts["advanced_processing"].extend(random.sample(enhancements, min(2, len(enhancements))))
        
        return instruction_parts
    
    def _generate_verification_points(self, target: str, complexity_level: int, content_type: str) -> List[str]:
        """Generate comprehensive verification points scaled by complexity"""
        base_points = [
            f"Ensure all original {content_type} content is preserved in the {target}",
            f"Validate that {target} structure is properly formed and coherent",
            "Verify that all transformation rules were correctly applied"
        ]
        
        complexity_points = {
            2: ["Confirm all conditional operations were properly executed"],
            3: ["Verify genre/style transformation maintains factual accuracy", 
                "Confirm creative elements enhance rather than replace original content"],
            4: ["Validate consistency across all perspectives or sections",
                "Verify meaningful connections between different transformation layers",
                "Confirm logical flow and narrative coherence"],
            5: ["Verify philosophical coherence across all transformation levels",
                "Confirm meta-transformation reveals genuine insights",
                "Validate that recontextualization preserves essential truths"]
        }
        
        points = base_points.copy()
        for level in range(2, complexity_level + 1):
            if level in complexity_points:
                points.extend(complexity_points[level])
        
        return points


class EnhancedContentGenerator:
    """Generates diverse content spanning creative to technical spectrum with significantly enhanced variations"""
    
    def __init__(self, llm: BenchmarkLLM, topic: Optional[str], logger: logging.Logger):
        self.llm = llm
        self.topic = topic
        self.logger = logger
        
        # Enhanced base prompts remain the same as they work well
        self.base_prompts = {
            # Creative content types
            "poetry": [
                "Write a poem of 12-16 lines with vivid imagery and emotional depth about life, nature, or human experience.",
                "Create a sequence of 3-5 haikus exploring a unified theme with natural imagery.",
                "Compose a narrative poem that tells a complete story in verse form.",
                "Write free verse poetry with strong metaphorical language and rhythm."
            ],
            "short_story": [
                "Write a 300-400 word short story with clear characters, plot, and resolution.",
                "Create a flash fiction piece with dialogue and a surprising twist ending.",
                "Write a story that reveals character through conversation and action.",
                "Compose a story with rich atmospheric description and emotional depth."
            ],
            "song_lyrics": [
                "Write song lyrics with verse-chorus structure telling an emotional story.",
                "Create narrative ballad lyrics with rhyme and rhythm.",
                "Compose lyrics with strong imagery and metaphorical language.",
                "Write abstract lyrics with layered meaning and musical flow."
            ],
            "screenplay": [
                "Write a 2-3 scene screenplay excerpt with dialogue, action, and character development.",
                "Create a dramatic monologue with detailed stage directions.",
                "Compose a dialogue scene showing conflict between two characters.",
                "Write a screenplay scene with visual storytelling and emotional beats."
            ],
            "philosophical_essay": [
                "Write a philosophical exploration of consciousness, identity, or meaning.",
                "Compose an essay examining truth, knowledge, or reality.",
                "Create a dialogue between opposing philosophical viewpoints.",
                "Write an essay connecting abstract concepts to concrete examples."
            ],
            
            # Technical content types  
            "code": [
                "Write a Python class implementing a data structure with 4-5 methods including initialization, insertion, deletion, and search.",
                "Create a JavaScript module with async functions, error handling, and documentation.",
                "Write a complete Python script with classes, file operations, and exception handling.",
                "Generate a working code solution with algorithm implementation and optimization."
            ],
            "api_documentation": [
                "Create REST API documentation with 3-4 endpoints, parameters, request/response examples, and error codes.",
                "Write comprehensive API reference with authentication, rate limiting, and usage examples.",
                "Generate OpenAPI specification with detailed schemas, responses, and error handling.",
                "Create developer guide with API integration examples and best practices."
            ],
            "database_schema": [
                "Design a database schema with 4-5 related tables, primary keys, foreign keys, and constraints.",
                "Create data model with indexes, relationships, and business rule constraints.",
                "Generate schema documentation with table descriptions and relationship diagrams.",
                "Design normalized database structure with proper referential integrity."
            ],
            "system_logs": [
                "Generate realistic server logs with timestamps, log levels, and various system events.",
                "Create application logs showing errors, warnings, info, and debug messages.",
                "Generate network logs with connection details, traffic, and performance metrics.",
                "Create security audit logs with user actions, access attempts, and system changes."
            ],
            
            # Hybrid content types
            "interview_transcript": [
                "Create an interview transcript between a journalist and technical expert with natural conversation flow.",
                "Generate a podcast transcript with multiple speakers discussing industry trends.",
                "Write a Q&A session with detailed technical explanations and follow-up questions.",
                "Create an expert interview with insights, examples, and practical advice."
            ],
            "product_review": [
                "Write a detailed product review with pros, cons, technical analysis, and rating.",
                "Create a comparative review examining multiple similar products with recommendations.",
                "Generate user experience review with specific use cases and real-world testing.",
                "Write expert review with technical specifications and performance analysis."
            ],
            "tutorial": [
                "Create a step-by-step tutorial with clear instructions, prerequisites, and troubleshooting tips.",
                "Write a beginner-friendly guide with examples, common mistakes, and best practices.",
                "Generate an advanced tutorial with optimization tips and professional techniques.",
                "Create an interactive tutorial with exercises and checkpoint validations."
            ],
            "case_study": [
                "Write a business case study with problem analysis, solution implementation, and results.",
                "Create a technical case study examining architecture decisions and outcomes.",
                "Generate a research case study with methodology, findings, and conclusions.",
                "Write a project case study with challenges, solutions, and lessons learned."
            ],
            "scientific_paper": [
                "Create a research paper excerpt with methodology, results, and discussion sections.",
                "Write a scientific abstract with hypothesis, methods, findings, and conclusions.",
                "Generate a literature review section with citations and comparative analysis.",
                "Create an experimental design with controls, variables, and statistical methods."
            ],
            
            # Traditional content types (enhanced)
            "text": [
                "Write an analytical article examining a complex issue with multiple perspectives and evidence.",
                "Create an explanatory piece making a technical topic accessible to general audiences.",
                "Compose an opinion piece with strong argumentation and supporting examples.",
                "Write a narrative non-fiction piece with personal insights and broader implications."
            ],
            "data": [
                "Generate a realistic dataset with relationships, patterns, and statistical significance.",
                "Create time-series data showing trends, seasonality, and meaningful variations.",
                "Generate survey data with demographic variables and response distributions.",
                "Create experimental data with control groups, variables, and measurable outcomes."
            ],
            "configuration": [
                "Create a complex application configuration with environment-specific settings and comments.",
                "Generate infrastructure configuration with dependencies and deployment parameters.",
                "Write system configuration with security settings and performance tuning.",
                "Create deployment configuration with scaling, monitoring, and backup settings."
            ],
            "documentation": [
                "Write comprehensive user documentation with setup, usage, and troubleshooting sections.",
                "Create technical specification with architecture, protocols, and implementation details.",
                "Generate developer documentation with installation, configuration, and contribution guidelines.",
                "Write API documentation with interactive examples and comprehensive reference."
            ]
        }
        
        # Significantly enhanced creative variations with more sophisticated options
        self.creative_variations = [
            # Style and Voice Variations
            "use rich descriptive language with sophisticated vocabulary and varied sentence structures",
            "adopt distinctive narrative voice with consistent personality and unique perspective",
            "incorporate sophisticated dialogue with authentic character voices and realistic speech patterns",
            "use metaphorical and symbolic language with multiple layers of meaning",
            "employ varied rhetorical devices including alliteration, assonance, and rhythmic patterns",
            "create immersive atmospheric descriptions with sensory details and emotional resonance",
            
            # Structural and Format Variations
            "include numbered lists, bullet points, and structured elements with clear hierarchy",
            "add timestamps, dates, and chronological markers with proper formatting",
            "incorporate quotes, citations, and reference materials with accurate attribution",
            "organize with clear sections, headers, and navigational structure",
            "use specialized formatting with emphasis, highlighting, and visual organization",
            "create interactive elements with reader engagement and participation opportunities",
            
            # Technical and Professional Variations
            "include specific technical terminology with precise language and domain expertise",
            "add code snippets, examples, and implementation details with proper syntax",
            "incorporate data, statistics, and quantitative elements with accurate calculations",
            "include configuration settings, parameters, and specifications with detailed explanations",
            "use professional jargon appropriately with clear definitions and context",
            "add troubleshooting sections with common problems and detailed solutions",
            
            # Creative Enhancement Variations
            "add emotional resonance with human interest elements and personal connections",
            "include cultural references with contextual background and historical significance",
            "incorporate humor, irony, and rhetorical devices with appropriate tone",
            "use dramatic techniques with tension, pacing, and narrative momentum",
            "create educational value with learning objectives and knowledge transfer",
            "add inspirational elements with motivational language and empowering messages",
            
            # Content Depth Variations
            "provide multiple perspectives with diverse viewpoints and expert opinions",
            "include historical context with evolutionary development and background information",
            "add practical applications with real-world examples and implementation strategies",
            "incorporate theoretical frameworks with conceptual analysis and scholarly depth",
            "create comparative analysis with contrasting approaches and alternative methods",
            "build thematic connections with underlying patterns and universal principles",
            
            # Special Character and Formatting Variations
            "include special characters, symbols, and Unicode elements for enhanced visual appeal",
            "use ASCII art, diagrams, and visual representations where appropriate",
            "incorporate mathematical notation, scientific symbols, and technical characters",
            "add decorative elements like dividers, borders, and visual separators",
            "use typographical emphasis with bold, italic, and formatting variations",
            "include international characters, accents, and multilingual elements where relevant",
            
            # Interactive and Engagement Variations
            "create call-to-action elements with reader engagement and response opportunities",
            "include questions, prompts, and interactive elements for active participation",
            "add exercises, challenges, and practical application opportunities",
            "create assessment elements with self-evaluation and progress tracking",
            "include collaborative elements with community engagement and sharing opportunities",
            "build anticipation with cliffhangers, mysteries, and progressive revelation"
        ]
    
    def generate_content(self, content_type: str, max_retries: int = 3) -> Tuple[str, List[str], int]:
        """Generate diverse content with enhanced prompting and creative variations"""
        
        attempts = 0
        last_exception = None
        
        for attempt in range(max_retries):
            attempts += 1
            try:
                # Select 2-3 variations for more sophisticated content
                num_variations = random.randint(2, 3)
                selected_variations = random.sample(self.creative_variations, num_variations)
                
                self.logger.debug(f"Generating {content_type} content - Attempt {attempt + 1} with {num_variations} variations")
                
                # Build clean, direct prompt
                base_prompt = random.choice(self.base_prompts.get(content_type, [
                    f"Generate a comprehensive {content_type} example with realistic structure and detailed content."
                ]))
                
                # Simple topic integration
                if self.topic:
                    base_prompt = self._integrate_topic_simple(base_prompt, content_type, self.topic)
                
                # Add enhancement instructions
                enhancement_text = f" Make sure to {selected_variations[0]}."
                if len(selected_variations) > 1:
                    enhancement_text += f" Also {selected_variations[1]}."
                if len(selected_variations) > 2:
                    enhancement_text += f" Additionally, {selected_variations[2]}."
                
                final_prompt = base_prompt + enhancement_text
                
                if attempt == 0:
                    print(f"    🎨 Enhancements: {', '.join(selected_variations[:2])}{'...' if len(selected_variations) > 2 else ''}")
                    print(f"    📝 Content type: {content_type}")
                else:
                    print(f"    🔄 Retry {attempt}: Generating content...")
                
                # Simplified system prompt for clarity
                system_prompt = (
                    f"You are a professional {content_type} writer. Create high-quality, realistic content. "
                    f"Output only the requested {content_type} content without explanations or meta-commentary."
                )
                
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("user", final_prompt)
                ])
                
                if attempt == 0:
                    # Get model info for display
                    llm_info = self.llm.get_llm_info()
                    creative_model = llm_info['creative']['model'] or 'Local/Default'
                    print(f"    🤖 Generating {content_type} content... (model: {creative_model})")
                
                # Generate content
                prompt_input = prompt_template.format_messages()
                llm_response = self.llm.call_creative_llm(prompt_input)
                response = ThinkTagSkippingParser().parse(llm_response.content)
                
                # Validate response quality
                if not response or len(response.strip()) < 20:
                    raise ValueError("Generated content is too short or empty")
                
                # Check if response is just the prompt (common LLM error)
                if "Make sure to" in response or "Also" in response or len(response) < 100:
                    raise ValueError("LLM returned prompt instead of content")
                
                # Additional validation for specific content types
                if not self._validate_content_quality(response, content_type):
                    raise ValueError(f"Generated {content_type} content does not meet quality standards")
                
                self.logger.info(f"Generated {len(response)} chars of {content_type} content with variations: {selected_variations}")
                if attempt == 0:
                    print(f"    ✅ Generated {len(response)} chars with {num_variations} enhancements")
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
    
    def _integrate_topic_simple(self, base_prompt: str, content_type: str, topic: str) -> str:
        """Simple topic integration to avoid prompt confusion"""
        if content_type in ["poetry", "short_story", "song_lyrics", "philosophical_essay"]:
            return base_prompt.replace("about life, nature, or human experience", f"about {topic}")
        elif content_type in ["code", "api_documentation", "tutorial"]:
            return base_prompt + f" Focus on {topic} as the subject matter."
        else:
            return base_prompt + f" The content should relate to {topic}."
    
    def _validate_content_quality(self, content: str, content_type: str) -> bool:
        """Validate that generated content meets quality standards for its type"""
        content = content.strip()
        
        # Basic length requirements
        min_lengths = {
            "poetry": 80,
            "short_story": 250,
            "song_lyrics": 100,
            "code": 150,
            "scientific_paper": 200,
            "tutorial": 200,
            "api_documentation": 150
        }
        
        min_length = min_lengths.get(content_type, 100)
        if len(content) < min_length:
            return False
        
        return True