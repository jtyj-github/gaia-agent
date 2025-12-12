"""
Code Executor Agent.
Generates and executes code to solve problems.
"""

from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from tools.code_tools import run_python_code, run_bash_command
from utils import setup_logger

logger = setup_logger("code_executor")


class CodeExecutorAgent:
    """Agent for executing code to solve problems."""

    def __init__(self, llm, config: Dict):
        """
        Initialize code executor agent.

        Args:
            llm: Language model instance
            config: Configuration dictionary with agent settings
        """
        self.llm = llm
        self.config = config
        self.enable_python = config.get('enable_python', True)
        self.enable_bash = config.get('enable_bash', True)
        self.system_prompt = config.get('system_prompt', '')

    def execute(self, task: str, language: str = "python") -> Dict[str, Any]:
        """
        Execute code to solve a task.

        Args:
            task: Task description or question
            language: Programming language (python or bash)

        Returns:
            Dictionary with execution results
        """
        logger.info(f"Code executor processing task: {task[:100]}...")

        # Step 1: Generate code using LLM
        code = self._generate_code(task, language)

        if not code:
            logger.error("Failed to generate code")
            return {
                'success': False,
                'message': 'Failed to generate code',
                'code': '',
                'output': '',
                'interpretation': ''
            }

        # Step 2: Execute code
        if language.lower() == "python" and self.enable_python:
            logger.info("Executing Python code")
            result = run_python_code(code)
        elif language.lower() == "bash" and self.enable_bash:
            logger.info("Executing Bash command")
            result = run_bash_command(code)
        else:
            logger.error(f"Language {language} not enabled")
            return {
                'success': False,
                'message': f'Language {language} not enabled',
                'code': code,
                'output': '',
                'interpretation': ''
            }

        # Step 3: Interpret results
        interpretation = self._interpret_results(task, code, result)

        return {
            'success': result['success'],
            'code': code,
            'output': result['output'],
            'interpretation': interpretation,
            'exit_code': result['exit_code']
        }

    def _generate_code(self, task: str, language: str) -> str:
        """
        Generate code using LLM.

        Args:
            task: Task description
            language: Programming language

        Returns:
            Generated code as string
        """
        prompt = f"""Write {language} code to solve the following task:

Task: {task}

Requirements:
- Write only the code, no explanations
- Make it self-contained
- Print the final result clearly
- Handle errors gracefully

Code:
"""

        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]

            response = self.llm.invoke(messages)
            code = response.content if hasattr(response, 'content') else str(response)

            # Clean up code - remove markdown code blocks if present
            code = code.strip()
            if code.startswith('```'):
                lines = code.split('\n')
                # Remove first line (```python or similar) and last line (```)
                if lines[-1].strip() == '```':
                    lines = lines[1:-1]
                else:
                    lines = lines[1:]
                code = '\n'.join(lines)

            logger.info(f"Generated {language} code ({len(code)} chars)")
            return code

        except Exception as e:
            logger.error(f"Error generating code: {e}")
            return ""

    def _interpret_results(self, task: str, code: str, result: Dict) -> str:
        """
        Interpret code execution results using LLM.

        Args:
            task: Original task
            code: Executed code
            result: Execution result dictionary

        Returns:
            Interpretation of results
        """
        if not result['success']:
            return f"Code execution failed: {result['output']}"

        prompt = f"""Given the task and code execution output, provide the final answer:

Task: {task}

Code executed:
{code}

Output:
{result['output']}

What is the final answer to the task? Provide a clear, concise answer.
"""

        try:
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            interpretation = response.content if hasattr(response, 'content') else str(response)

            logger.info(f"Interpreted results: {interpretation[:100]}...")
            return interpretation

        except Exception as e:
            logger.error(f"Error interpreting results: {e}")
            # Return raw output if interpretation fails
            return result['output']
