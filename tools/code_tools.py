"""
Code execution tools using Docker for sandboxing.
Provides safe Python and Bash code execution.
"""

import docker
import tempfile
import os
from typing import Dict
from utils import setup_logger

logger = setup_logger("code_tools")


class CodeExecutor:
    """Execute code in Docker sandbox for security."""

    def __init__(self, timeout: int = 30):
        """
        Initialize code executor.

        Args:
            timeout: Execution timeout in seconds
        """
        self.timeout = timeout
        try:
            self.client = docker.from_env()
            logger.info("Docker client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            self.client = None

    def execute_python(self, code: str) -> Dict:
        """
        Execute Python code in sandbox.

        Args:
            code: Python code to execute

        Returns:
            Dictionary with success status, output, and exit code
        """
        if not self.client:
            return {
                'success': False,
                'output': 'Docker is not available',
                'exit_code': -1
            }

        logger.info("Executing Python code")

        try:
            # Create temp file with code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            # Run in Docker
            container = self.client.containers.run(
                image="python:3.11-slim",
                command=f"python {os.path.basename(temp_file)}",
                volumes={os.path.dirname(temp_file): {'bind': '/workspace', 'mode': 'rw'}},
                working_dir='/workspace',
                detach=True,
                mem_limit='512m',
                network_disabled=True
            )

            # Wait for completion
            result = container.wait(timeout=self.timeout)
            output = container.logs().decode('utf-8')

            # Cleanup
            container.remove()
            os.unlink(temp_file)

            logger.info(f"Python execution completed with exit code: {result['StatusCode']}")
            return {
                'success': result['StatusCode'] == 0,
                'output': output,
                'exit_code': result['StatusCode']
            }

        except Exception as e:
            logger.error(f"Python execution error: {e}")
            if 'temp_file' in locals() and os.path.exists(temp_file):
                os.unlink(temp_file)
            return {
                'success': False,
                'output': str(e),
                'exit_code': -1
            }

    def execute_bash(self, command: str) -> Dict:
        """
        Execute bash command in sandbox.

        Args:
            command: Bash command to execute

        Returns:
            Dictionary with success status, output, and exit code
        """
        if not self.client:
            return {
                'success': False,
                'output': 'Docker is not available',
                'exit_code': -1
            }

        logger.info(f"Executing bash: {command}")

        try:
            container = self.client.containers.run(
                image="ubuntu:22.04",
                command=["bash", "-c", command],
                detach=True,
                mem_limit='512m',
                network_disabled=True
            )

            result = container.wait(timeout=self.timeout)
            output = container.logs().decode('utf-8')

            container.remove()

            logger.info(f"Bash execution completed with exit code: {result['StatusCode']}")
            return {
                'success': result['StatusCode'] == 0,
                'output': output,
                'exit_code': result['StatusCode']
            }

        except Exception as e:
            logger.error(f"Bash execution error: {e}")
            return {
                'success': False,
                'output': str(e),
                'exit_code': -1
            }


# Global executor instance
executor = CodeExecutor()


def run_python_code(code: str) -> Dict:
    """
    Convenience function for Python execution.

    Args:
        code: Python code to execute

    Returns:
        Execution result dictionary
    """
    return executor.execute_python(code)


def run_bash_command(command: str) -> Dict:
    """
    Convenience function for bash execution.

    Args:
        command: Bash command to execute

    Returns:
        Execution result dictionary
    """
    return executor.execute_bash(command)
