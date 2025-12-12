"""
Answer Validator Agent.
Normalizes and validates answers for GAIA benchmark evaluation.
Implements GAIA-specific answer format requirements.
"""

import re
from typing import Any, Dict
from utils import setup_logger

logger = setup_logger("validator")


class AnswerValidator:
    """Validate and normalize answers for GAIA evaluation."""

    def __init__(self, config: Dict = None):
        """
        Initialize validator with configuration.

        Args:
            config: Configuration dictionary with normalization rules and patterns
        """
        self.config = config or {}
        self.answer_pattern = self.config.get(
            'answer_extraction_pattern',
            r'FINAL ANSWER:\s*(.+)'
        )

    def extract_final_answer(self, text: str) -> str:
        """
        Extract final answer from agent response using GAIA format.
        Looks for "FINAL ANSWER: [answer]" pattern.

        Args:
            text: Full response text

        Returns:
            Extracted final answer
        """
        # Try to extract using GAIA pattern first
        match = re.search(self.answer_pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            logger.info(f"Extracted FINAL ANSWER: {answer[:100]}...")
            return answer

        # Fallback: look for other common patterns
        fallback_patterns = [
            r'answer[:\s]+(.+)',
            r'the answer is[:\s]+(.+)',
            r'result[:\s]+(.+)',
        ]

        for pattern in fallback_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                logger.warning(f"Used fallback pattern: {pattern}")
                return match.group(1).strip()

        # If no pattern found, return last non-empty line
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            logger.warning("No FINAL ANSWER pattern found, using last line")
            return lines[-1]

        return text.strip()

    def normalize_for_gaia(self, answer: str, answer_type: str = None) -> str:
        """
        Normalize answer according to GAIA requirements.

        GAIA Rules:
        - Numbers: No commas, no units (unless specified)
        - Strings: No articles, no abbreviations, digits in plain text
        - Lists: Comma-separated, apply rules per element

        Args:
            answer: Raw answer string
            answer_type: Type hint (number, string, list)

        Returns:
            Normalized answer string
        """
        if not answer:
            return ""

        answer = str(answer).strip()

        # Remove common articles if it's a string answer
        if answer_type == "string" or answer_type is None:
            # Remove leading articles
            answer = re.sub(r'^\b(the|a|an)\b\s+', '', answer, flags=re.IGNORECASE)

        # For numbers, remove commas and clean up
        if answer_type == "number" or self._looks_like_number(answer):
            # Remove commas from numbers
            answer = answer.replace(',', '')
            # Remove trailing units/symbols unless they look intentional
            answer = re.sub(r'(\d)\s*[\$%]$', r'\1', answer)

        return answer.strip()

    def _looks_like_number(self, text: str) -> bool:
        """Check if text looks like a number."""
        # Remove commas and check if it's numeric
        cleaned = text.replace(',', '').replace('$', '').replace('%', '').strip()
        try:
            float(cleaned)
            return True
        except ValueError:
            return False

    def normalize(self, answer: str) -> str:
        """
        Basic normalization for comparison (used in evaluation).

        Args:
            answer: Raw answer string

        Returns:
            Normalized answer string
        """
        if not answer:
            return ""

        # Convert to string
        answer = str(answer).strip()

        # Lowercase
        answer = answer.lower()

        # Remove extra whitespace
        answer = re.sub(r'\s+', ' ', answer)

        # Remove common punctuation at the end
        answer = re.sub(r'[.,;:!?]$', '', answer)

        return answer

    def validate_format(self, answer: str, expected_format: str) -> bool:
        """
        Check if answer matches expected format.

        Args:
            answer: Answer to validate
            expected_format: Expected format (number, date, yes_no)

        Returns:
            True if format matches, False otherwise
        """
        if expected_format == "number":
            return self._looks_like_number(answer)

        elif expected_format == "date":
            # Check common date formats
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
            ]
            return any(re.search(pattern, answer) for pattern in date_patterns)

        elif expected_format == "yes_no":
            return answer.strip().lower() in ['yes', 'no', 'true', 'false']

        elif expected_format == "list":
            return ',' in answer

        return True  # Default: accept any format

    def validate(self, answer: Any, expected_format: str = None) -> Dict[str, Any]:
        """
        Validate answer and return structured result.

        Args:
            answer: Answer to validate
            expected_format: Expected format (optional)

        Returns:
            Dictionary with validation results
        """
        answer_str = str(answer).strip()

        # Normalize for GAIA
        gaia_normalized = self.normalize_for_gaia(answer_str, expected_format)

        result = {
            'original': answer_str,
            'normalized': self.normalize(answer_str),
            'gaia_format': gaia_normalized,
            'valid': True,
            'format_match': True
        }

        if expected_format:
            result['format_match'] = self.validate_format(answer_str, expected_format)

        logger.info(f"Validated: '{answer_str[:50]}...' -> GAIA: '{gaia_normalized}'")
        return result


# Global validator instance
validator = AnswerValidator()
