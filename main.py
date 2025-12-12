"""
Main application entry point for GAIA Multi-Agent System.
Initializes all components and provides interface for question answering.
"""

import os
import yaml
from dotenv import load_dotenv
from pathlib import Path
from utils import setup_logger

# Import agents
from agents.orchestrator import Orchestrator
from agents.web_agent import WebAgent
from agents.code_executor import CodeExecutorAgent
from agents.file_handler import FileHandlerAgent
from agents.validator import AnswerValidator

logger = setup_logger("main")


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    logger.info(f"Loading config: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def initialize_llm(config: dict):
    """
    Initialize LLM based on configuration.

    Args:
        config: Model configuration dictionary

    Returns:
        Initialized LLM instance
    """
    provider = os.getenv('MODEL_PROVIDER', config['model']['provider'])
    logger.info(f"Initializing LLM with provider: {provider}")

    if provider == 'ollama':
        from langchain_community.llms import Ollama

        llm_config = config['model']['ollama']
        llm = Ollama(
            base_url=llm_config['base_url'],
            model=llm_config['model'],
            temperature=llm_config['temperature']
        )
        logger.info(f"Ollama initialized with model: {llm_config['model']}")
        return llm

    elif provider == 'huggingface':
        from langchain_community.llms import HuggingFaceHub

        llm_config = config['model']['huggingface']
        hf_token = os.getenv('HF_TOKEN')

        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables")

        llm = HuggingFaceHub(
            repo_id=llm_config['model'],
            huggingfacehub_api_token=hf_token,
            model_kwargs={
                'temperature': llm_config['temperature'],
                'max_length': llm_config['max_tokens']
            }
        )
        logger.info(f"HuggingFace initialized with model: {llm_config['model']}")
        return llm

    else:
        raise ValueError(f"Unknown provider: {provider}")


def initialize_agents(llm, agent_config: dict) -> dict:
    """
    Initialize all agents.

    Args:
        llm: Language model instance
        agent_config: Agent configuration dictionary

    Returns:
        Dictionary of initialized agents
    """
    logger.info("Initializing agents...")

    agents = {
        'web_agent': WebAgent(llm, agent_config['web_agent']),
        'code_executor': CodeExecutorAgent(llm, agent_config['code_executor']),
        'file_handler': FileHandlerAgent(llm, agent_config['file_handler']),
        'validator': AnswerValidator(agent_config['validator'])
    }

    logger.info(f"Initialized {len(agents)} agents")
    return agents


def main():
    """Main application entry point."""

    # Load environment variables
    load_dotenv()

    logger.info("="*60)
    logger.info("Starting GAIA Multi-Agent System")
    logger.info("="*60)

    try:
        # Load configurations
        logger.info("Loading configurations...")
        model_config = load_config('config/model_config.yaml')
        agent_config = load_config('config/agent_config.yaml')
        tool_config = load_config('config/tool_config.yaml')

        # Get GAIA system prompt
        gaia_prompt = agent_config.get('system_prompt', '')

        # Initialize LLM
        logger.info("Initializing LLM...")
        llm = initialize_llm(model_config)

        # Initialize agents
        logger.info("Initializing agents...")
        agents = initialize_agents(llm, agent_config)

        # Create orchestrator
        logger.info("Creating orchestrator...")
        orchestrator = Orchestrator(
            llm=llm,
            agents=agents,
            config=agent_config['orchestrator'],
            gaia_prompt=gaia_prompt
        )

        logger.info("System initialized successfully!")
        logger.info("="*60)

        # Test with a simple question
        test_question = "Based on historical data, when will world war 3 occur, if it happens?"
        logger.info(f"\nTesting with question: {test_question}")

        result = orchestrator.run(test_question)

        # Extract final answer using validator
        validator = agents['validator']
        final_answer = validator.extract_final_answer(result['answer'])

        logger.info(f"\nFull Response:\n{result['answer']}")
        logger.info(f"\nExtracted Answer: {final_answer}")
        logger.info(f"\nReasoning Trace:")
        for i, step in enumerate(result['reasoning_trace'], 1):
            logger.info(f"  {i}. {step}")
        logger.info(f"\nCompleted in {result['steps']} steps")

        print("\n" + "="*60)
        print("GAIA Multi-Agent System - Test Complete")
        print("="*60)
        print(f"\nQuestion: {test_question}")
        print(f"\nFull Response:\n{result['answer']}")
        print(f"\nExtracted Answer: {final_answer}")
        print(f"\nSteps taken: {result['steps']}")
        print("="*60)

        return orchestrator, agents

    except Exception as e:
        logger.error(f"Error during initialization: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    orchestrator, agents = main()
