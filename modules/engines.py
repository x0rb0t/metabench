"""
Transformation and verification engines - FIXED VERSION.
"""

import json
import logging
import time
import statistics
from typing import List, Dict, Any, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from .llm import BenchmarkLLM
from .parsers import ThinkTagSkippingParser


class TransformationEngine:
    """Applies transformation instructions to content with retry logic"""
    
    def __init__(self, llm: BenchmarkLLM, logger: logging.Logger):
        self.llm = llm
        self.logger = logger
    
    def apply_transformation(self, content: str, instruction: Dict[str, Any], max_retries: int = 3) -> Tuple[str, int]:
        """Apply transformation instruction to content with retry logic"""
        
        attempts = 0
        last_exception = None
        instruction_text = self._format_instruction(instruction)
        
        for attempt in range(max_retries):
            attempts += 1
            try:
                self.logger.debug(f"Applying transformation - Attempt {attempt + 1}: {instruction['task']}")
                self.logger.debug(f"Instruction length: {len(instruction_text)} chars")
                
                if attempt == 0:
                    # Get model info for display
                    llm_info = self.llm.get_llm_info()
                    transform_model = llm_info['transform']['model'] or 'Local/Default'
                    print(f"    📝 Transformation instruction formatted ({len(instruction_text)} chars)")
                    print(f"    🎯 Task: {instruction['task']}")
                    print(f"    📋 Rules: {len(instruction['basic_rules'])} basic, {len(instruction.get('conditional_operations', []))} conditional")
                    print(f"    ⚙️  Invoking transformation chain... (model: {transform_model})")
                else:
                    print(f"    🔄 Retry {attempt}: Applying transformation...")
                
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", 
                     "You are a precise transformation engine. Apply the transformation instructions exactly. "
                     "Follow ALL rules and requirements. Output only the transformed content without explanations."),
                    ("user", 
                     "Original content:\n{content}\n\n"
                     "Transformation instructions:\n{instruction}\n\n"
                     "Apply the transformation and output the result:")
                ])
                
                # Use the retry-wrapped LLM call
                prompt_input = prompt_template.format_messages(
                    content=content,
                    instruction=instruction_text
                )
                llm_response = self.llm.call_transform_llm(prompt_input)
                response = ThinkTagSkippingParser().parse(llm_response.content)
                
                # Validate response
                if not response or len(response.strip()) < 5:
                    raise ValueError("Transformed content is too short or empty")
                
                self.logger.info(f"Transformation completed: {len(content)} → {len(response)} chars")
                if attempt == 0:
                    print(f"    ✅ Transformation completed ({len(response)} chars output)")
                else:
                    print(f"    ✅ Transformation completed successfully on attempt {attempt + 1}")
                
                return response, attempts
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Transformation attempt {attempt + 1} failed: {e}")
                if attempt == 0:
                    print(f"    ⚠️  Transformation failed: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.info(f"Retrying transformation in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        # All retries exhausted
        self.logger.error(f"All {max_retries} transformation attempts failed. Last error: {last_exception}")
        print(f"    ❌ All transformation attempts failed")
        raise last_exception
    
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
    """Verifies and scores transformation results with improved logic, retry, and multiple attempts - FIXED VERSION"""
    
    def __init__(self, llm: BenchmarkLLM, logger: logging.Logger):
        self.llm = llm
        self.logger = logger
    
    def verify_transformation(self, 
                            original_content: str,
                            instruction: Dict[str, Any],
                            transformed_content: str,
                            max_retries: int = 3,
                            verification_attempts: int = 1,
                            aggregation_method: str = 'avg') -> Tuple[float, float, Dict[str, float], int, List[float]]:
        """Verify transformation with multiple attempts and return aggregated scores"""
        
        instruction_text = self._format_verification_criteria(instruction)
        requirements_list = self._extract_requirements_list(instruction)
        applicable_categories = self._get_applicable_categories(instruction)
        
        self.logger.debug(f"Verifying transformation with {len(requirements_list)} requirements")
        self.logger.debug(f"Applicable categories: {applicable_categories}")
        self.logger.debug(f"Verification attempts: {verification_attempts}, Aggregation: {aggregation_method}")
        
        # Get model info for display
        llm_info = self.llm.get_llm_info()
        verification_model = llm_info['verification']['model'] or 'Local/Default'
        
        print(f"    🔍 Verification criteria: {len(instruction_text)} chars")
        print(f"    📊 Comparing {len(original_content)} → {len(transformed_content)} chars")
        print(f"    📋 Requirements to verify: {len(requirements_list)} items")
        print(f"    🎯 Applicable categories: {len(applicable_categories)}")
        print(f"    🔄 Verification attempts: {verification_attempts} ({aggregation_method}) (model: {verification_model})")
        
        all_verification_results = []
        total_attempts = 0
        
        for verification_attempt in range(verification_attempts):
            if verification_attempts > 1:
                print(f"      🔍 Verification attempt {verification_attempt + 1}/{verification_attempts}")
            
            result, attempts = self._single_verification_attempt(
                original_content, instruction_text, transformed_content, 
                requirements_list, applicable_categories, max_retries
            )
            
            all_verification_results.append(result)
            total_attempts += attempts
        
        # Aggregate results
        aggregated_result = self._aggregate_verification_results(all_verification_results, aggregation_method)
        
        quality_score = aggregated_result['quality_score']
        completion_rate = aggregated_result['completion_rate']
        specific_scores = aggregated_result['specific_scores']
        
        # Extract all quality scores for tracking
        all_quality_scores = [result['quality_score'] for result in all_verification_results]
        
        self.logger.info(f"Verification complete after {verification_attempts} attempts:")
        self.logger.info(f"  Quality scores: {all_quality_scores}")
        self.logger.info(f"  Aggregated ({aggregation_method}): Quality={quality_score:.2f}, Completion={completion_rate:.2%}")
        self.logger.info(f"  Specific scores: {specific_scores}")
        
        if verification_attempts > 1:
            score_range = f"{min(all_quality_scores):.2f}-{max(all_quality_scores):.2f}"
            print(f"    📈 Verification complete: {verification_attempts} attempts, scores {score_range}")
            print(f"    🎯 Aggregated ({aggregation_method}): Quality={quality_score:.2f}/10, Completion={completion_rate:.2%}")
        else:
            print(f"    ✅ Verification complete: Quality={quality_score:.2f}/10, Completion={completion_rate:.2%}")
        
        # Log each specific score with applicability
        for category, score in specific_scores.items():
            if category in applicable_categories:
                if score is not None:
                    print(f"      🎯 {category}: {score:.2f}/10")
                else:
                    print(f"      ⚠️  {category}: N/A (null score)")
            else:
                print(f"      ⚫ {category}: N/A (not applicable)")
        
        return quality_score, completion_rate, specific_scores, total_attempts, all_quality_scores
    
    def _single_verification_attempt(self, 
                                   original_content: str,
                                   instruction_text: str,
                                   transformed_content: str,
                                   requirements_list: List[str],
                                   applicable_categories: List[str],
                                   max_retries: int) -> Tuple[Dict[str, Any], int]:
        """Perform a single verification attempt with retry logic - FIXED VERSION"""
        
        attempts = 0
        last_exception = None
        
        # Simplified verification prompt to avoid JSON parsing issues
        verification_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a strict verification engine. Analyze transformation results objectively and return valid JSON only. "
             "Score applicable categories from 0-10. Use null for non-applicable categories. "
             "Ensure all JSON is properly formatted with no extra text."),
            ("user",
             "Original content:\n{original}\n\n"
             "Transformation instructions:\n{instruction}\n\n" 
             "Transformed result:\n{transformed}\n\n"
             "Requirements to evaluate:\n{requirements}\n\n"
             "Applicable categories: {categories}\n\n"
             "Return JSON with this EXACT structure (no additional text):\n"
             "{{\n"
             '  "specific_scores": {{\n'
             '    "basic_rules_score": <number 0-10 or null>,\n'
             '    "conditional_operations_score": <number 0-10 or null>,\n'
             '    "advanced_processing_score": <number 0-10 or null>,\n'
             '    "verification_requirements_score": <number 0-10>,\n'
             '    "data_preservation_score": <number 0-10>,\n'
             '    "format_compliance_score": <number 0-10>\n'
             '  }},\n'
             '  "quality_score": <number 0-10>,\n'
             '  "instruction_completion": <number 0-1>,\n'
             '  "feedback": "<brief explanation>"\n'
             "}}")
        ])
        
        for attempt in range(max_retries):
            attempts += 1
            try:
                if attempt > 0:
                    print(f"        🔄 Retry {attempt}: Running verification...")
                
                # Use the retry-wrapped LLM call
                prompt_input = verification_prompt.format_messages(
                    original=original_content,
                    instruction=instruction_text,
                    transformed=transformed_content,
                    requirements="\n".join([f"- {req}" for req in requirements_list]),
                    categories=", ".join(applicable_categories)
                )
                llm_response = self.llm.call_verification_llm(prompt_input)
                json_text = ThinkTagSkippingParser().parse(llm_response.content)
                
                # Clean JSON text to avoid parsing issues
                json_text = self._clean_json_text(json_text)
                
                try:
                    result = JsonOutputParser().parse(json_text)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"JSON parsing failed, attempting to extract JSON from response: {e}")
                    result = self._extract_json_from_text(json_text)
                
                # Process specific scores with proper null handling - FIXED
                specific_scores = self._process_specific_scores_fixed(
                    result.get('specific_scores', {}), 
                    applicable_categories
                )
                
                # Extract and validate overall scores
                quality_score = self._validate_float_score(result.get('quality_score', 0), 0, 10, "quality_score")
                completion_rate = self._validate_float_score(result.get('instruction_completion', 0), 0, 1, "instruction_completion")
                
                # Calculate average only from applicable categories - FIXED
                applicable_scores = [score for cat, score in specific_scores.items() 
                                   if cat in applicable_categories and score is not None]
                avg_specific = sum(applicable_scores) / len(applicable_scores) if applicable_scores else 0
                
                feedback = result.get('feedback', 'No feedback provided')
                
                return {
                    'quality_score': quality_score,
                    'completion_rate': completion_rate,
                    'specific_scores': specific_scores,
                    'feedback': feedback,
                    'avg_specific': avg_specific
                }, attempts
                
            except (ConnectionError, TimeoutError) as e:
                last_exception = e
                self.logger.warning(f"Network error during verification attempt {attempt + 1}: {e}")
                if attempt == 0:
                    print(f"    🌐 Network error during verification: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.info(f"Retrying verification in {wait_time} seconds...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Verification attempt {attempt + 1} failed: {e}")
                if attempt == 0:
                    print(f"    ❌ Verification error: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.info(f"Retrying verification in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        # All retries exhausted - return default scores
        self.logger.error(f"All {max_retries} verification attempts failed. Last error: {last_exception}")
        print(f"    ❌ All verification attempts failed, using default scores")
        
        # Create safe default scores for applicable categories
        default_specific_scores = {}
        for category in ['basic_rules_score', 'conditional_operations_score', 'advanced_processing_score',
                        'verification_requirements_score', 'data_preservation_score', 'format_compliance_score']:
            if category in applicable_categories:
                default_specific_scores[category] = 0.0
            else:
                default_specific_scores[category] = None
        
        return {
            'quality_score': 0.0,
            'completion_rate': 0.0,
            'specific_scores': default_specific_scores,
            'feedback': f'Verification failed: {last_exception}',
            'avg_specific': 0.0
        }, attempts
    
    def _clean_json_text(self, json_text: str) -> str:
        """Clean JSON text to remove common formatting issues"""
        # Remove code block markers
        json_text = json_text.replace("```json", "").replace("```", "")
        # Remove extra whitespace
        json_text = json_text.strip()
        # Find JSON object boundaries
        start = json_text.find('{')
        end = json_text.rfind('}') + 1
        if start >= 0 and end > start:
            json_text = json_text[start:end]
        return json_text
    
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text when normal parsing fails"""
        try:
            # Try to find JSON-like structure
            import re
            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Return minimal valid structure if all else fails
        return {
            "specific_scores": {
                "basic_rules_score": 0,
                "conditional_operations_score": None,
                "advanced_processing_score": None,
                "verification_requirements_score": 0,
                "data_preservation_score": 0,
                "format_compliance_score": 0
            },
            "quality_score": 0,
            "instruction_completion": 0,
            "feedback": "JSON parsing failed"
        }
    
    def _process_specific_scores_fixed(self, specific_scores: Dict[str, Any], applicable_categories: List[str]) -> Dict[str, float]:
        """Process specific scores with improved null handling - FIXED VERSION"""
        processed_scores = {}
        
        all_categories = [
            'basic_rules_score', 'conditional_operations_score', 'advanced_processing_score',
            'verification_requirements_score', 'data_preservation_score', 'format_compliance_score'
        ]
        
        for category in all_categories:
            score_value = specific_scores.get(category)
            
            if category in applicable_categories:
                # This category should be scored
                if score_value is None or score_value == "null":
                    self.logger.warning(f"Applicable category {category} was null, using 0.0")
                    processed_scores[category] = 0.0
                else:
                    try:
                        processed_scores[category] = self._validate_float_score(score_value, 0, 10, category)
                    except (ValueError, TypeError):
                        self.logger.warning(f"Invalid {category} value '{score_value}', using 0.0")
                        processed_scores[category] = 0.0
            else:
                # This category is not applicable
                processed_scores[category] = None
        
        return processed_scores
    
    def _aggregate_verification_results(self, results: List[Dict[str, Any]], method: str) -> Dict[str, Any]:
        """Aggregate multiple verification results using specified method"""
        
        if not results:
            return {
                'quality_score': 0.0,
                'completion_rate': 0.0,
                'specific_scores': {},
                'feedback': 'No verification results to aggregate'
            }
        
        quality_scores = [r['quality_score'] for r in results]
        completion_rates = [r['completion_rate'] for r in results]
        
        # Aggregate main scores
        if method == 'best':
            aggregated_quality = max(quality_scores)
            best_idx = quality_scores.index(aggregated_quality)
            aggregated_completion = completion_rates[best_idx]
            aggregated_specific = results[best_idx]['specific_scores']
            feedback = f"Best of {len(results)} attempts: {results[best_idx]['feedback']}"
        elif method == 'worst':
            aggregated_quality = min(quality_scores)
            worst_idx = quality_scores.index(aggregated_quality)
            aggregated_completion = completion_rates[worst_idx]
            aggregated_specific = results[worst_idx]['specific_scores']
            feedback = f"Worst of {len(results)} attempts: {results[worst_idx]['feedback']}"
        else:  # avg
            aggregated_quality = statistics.mean(quality_scores)
            aggregated_completion = statistics.mean(completion_rates)
            
            # Average specific scores - FIXED
            all_categories = set()
            for result in results:
                all_categories.update(result['specific_scores'].keys())
            
            aggregated_specific = {}
            for category in all_categories:
                category_scores = [r['specific_scores'].get(category) for r in results 
                                if r['specific_scores'].get(category) is not None]
                if category_scores:
                    aggregated_specific[category] = statistics.mean(category_scores)
                else:
                    aggregated_specific[category] = None
            
            feedback = f"Average of {len(results)} attempts (scores: {quality_scores})"
        
        return {
            'quality_score': aggregated_quality,
            'completion_rate': aggregated_completion,
            'specific_scores': aggregated_specific,
            'feedback': feedback
        }
    
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
            if value is None or value == "null":
                self.logger.warning(f"{field_name} is null, using 0.0")
                return 0.0
                
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