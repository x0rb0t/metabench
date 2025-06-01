"""
Custom output parsers for LLM responses.
"""

import re
from langchain_core.output_parsers import BaseOutputParser


class ThinkTagSkippingParser(BaseOutputParser[str]):
    """Parses out <think> tags and returns the remaining text."""

    def parse(self, text: str) -> str:
        text_without_think_tags = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text_without_think_tags.strip()

    @property
    def _type(self) -> str:
        return "think_tag_skipping_parser"
