"""
File Handler Agent.
Processes various file types and extracts information.
"""

from typing import Dict, Any
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from tools.file_tools import process_file
from utils import setup_logger

logger = setup_logger("file_handler")


class FileHandlerAgent:
    """Agent for processing various file types."""

    def __init__(self, llm, config: Dict):
        """
        Initialize file handler agent.

        Args:
            llm: Language model instance
            config: Configuration dictionary with agent settings
        """
        self.llm = llm
        self.config = config
        self.supported_formats = config.get('supported_formats', [])
        self.system_prompt = config.get('system_prompt', '')

    def process(self, file_path: str, question: str) -> Dict[str, Any]:
        """
        Process file and answer question about it.

        Args:
            file_path: Path to file
            question: Question about the file

        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing file: {file_path}")

        # Check if file exists
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            return {
                'success': False,
                'message': f'File not found: {file_path}',
                'answer': ''
            }

        # Check if format is supported
        ext = Path(file_path).suffix.lower()[1:]  # Remove the dot
        if ext not in self.supported_formats:
            logger.error(f"Unsupported format: {ext}")
            return {
                'success': False,
                'message': f'Unsupported format: {ext}',
                'answer': ''
            }

        # Process file
        file_data = process_file(file_path)

        if 'error' in file_data:
            logger.error(f"Error processing file: {file_data['error']}")
            return {
                'success': False,
                'message': file_data['error'],
                'answer': ''
            }

        # Use LLM to answer question about file
        answer = self._answer_from_file_data(file_data, question)

        return {
            'success': True,
            'answer': answer,
            'file_type': file_data.get('type'),
            'file_data': file_data
        }

    def _answer_from_file_data(self, file_data: Dict, question: str) -> str:
        """
        Use LLM to answer question based on file data.

        Args:
            file_data: Processed file data
            question: Question about the file

        Returns:
            Answer string
        """
        # Format file data for LLM
        context = self._format_file_data(file_data)

        prompt = f"""Based on the following file content, answer the question:

File Type: {file_data.get('type')}

Content:
{context}

Question: {question}

Provide a clear, specific answer based on the file content.
"""

        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]

            response = self.llm.invoke(messages)
            answer = response.content if hasattr(response, 'content') else str(response)

            logger.info(f"Generated answer from file: {answer[:100]}...")
            return answer

        except Exception as e:
            logger.error(f"Error during LLM invocation: {e}")
            return f"Error processing file: {str(e)}"

    def _format_file_data(self, file_data: Dict) -> str:
        """
        Format file data for LLM consumption.

        Args:
            file_data: Processed file data

        Returns:
            Formatted context string
        """
        file_type = file_data.get('type')

        if file_type == 'pdf':
            # Limit PDF text to avoid token overflow
            content = file_data.get('content', '')[:5000]
            return f"PDF Text Content:\n{content}"

        elif file_type == 'excel':
            lines = []
            for sheet, data in file_data.get('content', {}).items():
                lines.append(f"\nSheet: {sheet}")
                lines.append(f"Columns: {data.get('columns')}")
                lines.append(f"Number of rows: {len(data.get('rows', []))}")
                lines.append(f"First 10 rows: {data.get('rows')[:10]}")
                if data.get('summary'):
                    lines.append(f"Summary statistics: {data.get('summary')}")
            return '\n'.join(lines)

        elif file_type == 'csv':
            content = file_data.get('content', {})
            lines = [
                f"Columns: {content.get('columns')}",
                f"Shape: {content.get('shape')}",
                f"First 20 rows: {content.get('rows')[:20]}"
            ]
            if content.get('summary'):
                lines.append(f"Summary statistics: {content.get('summary')}")
            return '\n'.join(lines)

        elif file_type == 'image':
            info = file_data.get('info', {})
            text = file_data.get('text', '')
            lines = [
                f"Image format: {info.get('format')}",
                f"Image size: {info.get('size')}",
                f"Image mode: {info.get('mode')}",
                f"Has text: {info.get('has_text')}"
            ]
            if text.strip():
                lines.append(f"\nExtracted text (OCR):\n{text}")
            return '\n'.join(lines)

        # Default: return string representation
        return str(file_data)
