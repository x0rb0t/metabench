"""
LLM wrapper and setup functionality.
"""

import logging
from langchain_openai import ChatOpenAI
from .config import BenchmarkConfig


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
