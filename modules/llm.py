"""
LLM wrapper and setup functionality.
"""

import logging
import time
from typing import Callable, Any, TypeVar
from langchain_openai import ChatOpenAI
from .config import BenchmarkConfig

T = TypeVar('T')


def retry_llm_call(llm, input_data, max_retries: int, logger: logging.Logger):
    """Execute LLM call with retry logic and exponential backoff"""
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            result = llm.invoke(input_data)
            if attempt > 0:
                logger.info(f"LLM call succeeded on attempt {attempt + 1}")
            return result
        except (ConnectionError, TimeoutError) as e:
            last_exception = e
            wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, etc.
            logger.warning(f"Network error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying LLM call in {wait_time} seconds...")
                time.sleep(wait_time)
        except Exception as e:
            # For non-network errors, don't retry
            logger.error(f"Non-retryable error in LLM call: {e}")
            raise e
    
    # All retries exhausted
    logger.error(f"All {max_retries} LLM call attempts failed. Last error: {last_exception}")
    raise last_exception


class BenchmarkLLM:
    """Wrapper for LLM with different temperature settings and base URLs for different tasks"""
    
    def __init__(self, config: BenchmarkConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.max_retries = config.max_retries
        
        self.logger.info(f"Initializing LLMs with different base URLs:")
        self.logger.info(f"  Creative LLM: {config.creative_base_url}")
        self.logger.info(f"  Verification LLM: {config.verification_base_url}")
        self.logger.info(f"  Transform LLM: {config.transform_base_url}")
        self.logger.info(f"Model configuration: {config.model_name or 'Local/Default'}")
        self.logger.info(f"Max retries: {config.max_retries}")
        
        # Initialize LLMs with different base URLs and temperatures
        self.creative_llm = ChatOpenAI(
            base_url=config.creative_base_url,
            temperature=config.creative_temperature,
            api_key=config.creative_api_key,
            model=config.creative_model or config.model_name,
            timeout=60,
            max_retries=0  # We handle retries manually
        )
        
        self.verification_llm = ChatOpenAI(
            base_url=config.verification_base_url,
            temperature=config.verification_temperature,
            api_key=config.verification_api_key,
            model=config.verification_model or config.model_name,
            timeout=60,
            max_retries=0  # We handle retries manually
        )
        
        self.transform_llm = ChatOpenAI(
            base_url=config.transform_base_url,
            temperature=config.transform_temperature,
            api_key=config.transform_api_key,
            model=config.transform_model or config.model_name,
            timeout=60,
            max_retries=0  # We handle retries manually
        )
        
        self.logger.debug("Initialized all LLMs successfully")
    
    def call_creative_llm(self, input_data):
        """Call creative LLM with retry logic"""
        return retry_llm_call(self.creative_llm, input_data, self.max_retries, self.logger)
    
    def call_verification_llm(self, input_data):
        """Call verification LLM with retry logic"""
        return retry_llm_call(self.verification_llm, input_data, self.max_retries, self.logger)
    
    def call_transform_llm(self, input_data):
        """Call transform LLM with retry logic"""
        return retry_llm_call(self.transform_llm, input_data, self.max_retries, self.logger)
    
    def get_llm_info(self) -> dict:
        """Get information about configured LLMs"""
        return {
            "creative": {
                "base_url": self.config.creative_base_url,
                "model": self.config.creative_model or self.config.model_name,
                "temperature": self.config.creative_temperature
            },
            "verification": {
                "base_url": self.config.verification_base_url,
                "model": self.config.verification_model or self.config.model_name,
                "temperature": self.config.verification_temperature
            },
            "transform": {
                "base_url": self.config.transform_base_url,
                "model": self.config.transform_model or self.config.model_name,
                "temperature": self.config.transform_temperature
            }
        }