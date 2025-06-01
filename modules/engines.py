"""
Transformation and verification engines.
"""

import json
import logging
from typing import List, Dict, Any, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from .llm import BenchmarkLLM
from .parsers import ThinkTagSkippingParser


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
