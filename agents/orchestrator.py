"""
Orchestrator Agent using LangGraph.
Coordinates specialized agents to answer complex questions.
"""

from typing import Dict, Any, List, TypedDict, Annotated
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from utils import setup_logger
import operator

logger = setup_logger("orchestrator")


class OrchestratorState(TypedDict):
    """State object for LangGraph workflow."""
    question: str
    file_path: str
    current_step: int
    max_steps: int
    agent_outputs: Annotated[Dict[str, Any], operator.or_]
    final_answer: str
    history: Annotated[List[Dict], operator.add]
    reasoning_trace: Annotated[List[str], operator.add]


class Orchestrator:
    """Main orchestrator using LangGraph supervisor pattern."""

    def __init__(self, llm, agents: Dict, config: Dict, gaia_prompt: str = None):
        """
        Initialize orchestrator.

        Args:
            llm: Language model instance
            agents: Dictionary of agent name -> agent instance
            config: Configuration dictionary
            gaia_prompt: GAIA system prompt for final answer generation
        """
        self.llm = llm
        self.agents = agents
        self.config = config
        self.max_iterations = config.get('max_iterations', 10)
        self.system_prompt = config.get('system_prompt', '')
        self.gaia_prompt = gaia_prompt or ""
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build LangGraph workflow.

        Returns:
            Compiled StateGraph
        """
        workflow = StateGraph(OrchestratorState)

        # Add nodes
        workflow.add_node("supervisor", self.supervisor_node)
        workflow.add_node("web_agent", self.web_agent_node)
        workflow.add_node("code_executor", self.code_executor_node)
        workflow.add_node("file_handler", self.file_handler_node)
        workflow.add_node("synthesize", self.synthesize_node)

        # Set entry point
        workflow.set_entry_point("supervisor")

        # Add conditional edges from supervisor
        workflow.add_conditional_edges(
            "supervisor",
            self.route_decision,
            {
                "web_agent": "web_agent",
                "code_executor": "code_executor",
                "file_handler": "file_handler",
                "synthesize": "synthesize",
                "end": END
            }
        )

        # All agents return to supervisor
        for agent in ["web_agent", "code_executor", "file_handler"]:
            workflow.add_edge(agent, "supervisor")

        # Synthesize ends the workflow
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    def supervisor_node(self, state: OrchestratorState) -> OrchestratorState:
        """
        Supervisor decides next action.

        Args:
            state: Current workflow state

        Returns:
            Updated state
        """
        logger.info(f"Supervisor iteration {state['current_step']}")

        # Check termination conditions
        if state['current_step'] >= state['max_steps']:
            logger.warning("Max iterations reached")
            state['reasoning_trace'].append(f"Step {state['current_step']}: Max iterations reached, synthesizing answer")
            return state

        if state['final_answer']:
            logger.info("Final answer already set")
            return state

        # First iteration: decide whether to try direct answer or use tools
        # For GAIA benchmark, most questions require tools, so be conservative about direct answers
        if state['current_step'] == 0:
            # Only try direct answer for very simple questions
            # Check if file is attached - if so, always delegate to file_handler
            if state['file_path']:
                logger.info("File attached - skipping direct answer attempt")
            else:
                logger.info("First iteration: checking if direct answer is possible")
                needs_tools = self._check_if_tools_needed(state)
                if not needs_tools:
                    logger.info("Attempting direct answer")
                    direct_answer = self._try_direct_answer(state)
                    if direct_answer and not direct_answer.startswith("NEED_TOOLS"):
                        logger.info("Question answered directly without agent delegation")
                        state['final_answer'] = direct_answer
                        state['reasoning_trace'].append("Step 0: Answered directly using general knowledge")
                        return state
                else:
                    logger.info("Tools needed - skipping direct answer")

        # Make decision
        decision = self._make_decision(state)

        state['history'].append({
            'step': state['current_step'],
            'decision': decision
        })
        state['reasoning_trace'].append(f"Step {state['current_step']}: Decided to use {decision}")
        state['current_step'] += 1

        return state

    def _check_if_tools_needed(self, state: OrchestratorState) -> bool:
        """
        Quick check if question obviously requires tools.

        Args:
            state: Current workflow state

        Returns:
            True if tools are clearly needed, False otherwise
        """
        question_lower = state['question'].lower()

        # Keywords that suggest tools are needed
        tool_keywords = [
            # Web/URL related
            'arxiv', 'wikipedia', 'website', 'url', 'link', 'article', 'paper',
            'published', 'author', 'journal', 'doi',
            # Data/file related
            'file', 'attached', 'document', 'spreadsheet', 'csv', 'excel',
            # Current/recent data
            'latest', 'recent', 'current', '2020', '2021', '2022', '2023', '2024', '2025',
            'today', 'yesterday', 'this year', 'last year',
            # Complex calculations
            'calculate', 'compute', 'sum of', 'average of', 'total of',
            # Specific lookups
            'find the', 'search for', 'look up', 'what is the name of',
            'how many', 'list all', 'count the'
        ]

        return any(keyword in question_lower for keyword in tool_keywords)

    def _try_direct_answer(self, state: OrchestratorState) -> str:
        """
        Try to answer the question directly without delegating to agents.

        Args:
            state: Current workflow state

        Returns:
            Direct answer string if possible, empty string if agents needed
        """
        prompt = f"""Analyze this question and determine if you can answer it directly, or if you need specialized tools.

Question: {state['question']}
File attached: {state['file_path'] is not None and state['file_path'] != ''}

Determine if this question requires:
1. Web search (current events, recent data, specific URLs, real-time information)
2. Code execution (complex calculations, data processing, algorithms)
3. File processing (analyzing attached files)

If the question is:
- General knowledge (history, geography, science facts, etc.) → Answer directly
- Simple calculation (basic math) → Answer directly
- Requires current/recent data → Needs web search
- Requires file analysis → Needs file processing
- Requires complex computation → Needs code execution

If you can answer directly, provide your answer in the GAIA format (ending with "FINAL ANSWER: [answer]").
If you need specialized tools, respond with only: "NEED_TOOLS: [tool_type]" where tool_type is web_search, code_execution, or file_processing.

Your response:"""

        try:
            messages = [
                SystemMessage(content=self.gaia_prompt),
                HumanMessage(content=prompt)
            ]

            response = self.llm.invoke(messages)
            answer = response.content if hasattr(response, 'content') else str(response)
            answer = answer.strip()

            # Check if tools are needed
            if answer.startswith("NEED_TOOLS:"):
                logger.info(f"Direct answer not possible: {answer}")
                return ""

            # Return the direct answer
            logger.info(f"Direct answer provided: {answer[:100]}...")
            return answer

        except Exception as e:
            logger.error(f"Error in direct answer attempt: {e}")
            return ""  # Fall back to agent delegation

    def _make_decision(self, state: OrchestratorState) -> str:
        """
        Decide which agent to use next.

        Args:
            state: Current workflow state

        Returns:
            Decision string (agent name or action)
        """
        # Check if we should automatically synthesize
        # If we have any successful agent outputs, and we've made multiple attempts, synthesize
        successful_agents = [k for k, v in state['agent_outputs'].items() if v.get('success')]
        if successful_agents and state['current_step'] > len(successful_agents):
            logger.info(f"Auto-synthesizing: have {len(successful_agents)} successful outputs after {state['current_step']} steps")
            return 'synthesize'

        context = self._format_history(state)

        prompt = f"""You are coordinating agents to answer a question. Analyze the question and decide the next action.

Question: {state['question']}
File attached: {state['file_path'] is not None and state['file_path'] != ''}

Available agents:
- web_agent: Search web and browse pages for information
- code_executor: Execute Python/bash code for calculations or data processing
- file_handler: Process files (PDF, Excel, images, etc.)
- synthesize: Combine all gathered information into final answer

Previous actions:
{context}

Agent outputs so far:
{self._format_agent_outputs(state['agent_outputs'])}

IMPORTANT: Respond with ONLY a single word from this list:
web_agent, code_executor, file_handler, synthesize, end

Rules:
- If an agent has already been called successfully, do NOT call it again
- If you have any information from agents, choose "synthesize"
- Only call web_agent if you have NO information yet
- Only call code_executor if you need calculations AND haven't called it yet
- Only call file_handler if there's a file AND you haven't called it yet
- If an agent was called multiple times, always choose "synthesize"

Your decision (one word only):"""

        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]

            response = self.llm.invoke(messages)
            decision_text = response.content if hasattr(response, 'content') else str(response)
            decision_text = decision_text.strip().lower()

            # Extract just the decision word (first line or word)
            # Handle cases where LLM adds explanation after the decision
            decision = decision_text.split('\n')[0].split()[0].strip()

            # Validate decision
            valid_decisions = ['web_agent', 'code_executor', 'file_handler', 'synthesize', 'end']
            if decision not in valid_decisions:
                logger.warning(f"Invalid decision '{decision}' from response '{decision_text[:100]}...', defaulting to 'synthesize'")
                decision = 'synthesize'  # Changed default to synthesize instead of web_agent

            logger.info(f"Supervisor decision: {decision}")
            return decision

        except Exception as e:
            logger.error(f"Error making decision: {e}")
            return 'synthesize'  # Safe fallback

    def web_agent_node(self, state: OrchestratorState) -> OrchestratorState:
        """Execute web agent."""
        logger.info("Executing web agent")
        result = self.agents['web_agent'].search_and_extract(state['question'])
        state['agent_outputs']['web_agent'] = result
        state['reasoning_trace'].append(f"Web search: Found {len(result.get('sources', []))} sources")
        return state

    def code_executor_node(self, state: OrchestratorState) -> OrchestratorState:
        """Execute code executor."""
        logger.info("Executing code executor")
        result = self.agents['code_executor'].execute(state['question'])
        state['agent_outputs']['code_executor'] = result
        state['reasoning_trace'].append(f"Code execution: {'Success' if result.get('success') else 'Failed'}")
        return state

    def file_handler_node(self, state: OrchestratorState) -> OrchestratorState:
        """Execute file handler."""
        if state['file_path']:
            logger.info(f"Executing file handler for: {state['file_path']}")
            result = self.agents['file_handler'].process(state['file_path'], state['question'])
            state['agent_outputs']['file_handler'] = result
            state['reasoning_trace'].append(f"File processing: {result.get('file_type', 'unknown')} file processed")
        else:
            logger.warning("File handler called but no file path provided")
            state['reasoning_trace'].append("File handler: No file to process")
        return state

    def synthesize_node(self, state: OrchestratorState) -> OrchestratorState:
        """Synthesize final answer from all agent outputs."""
        logger.info("Synthesizing final answer")
        final_answer = self._synthesize_answer(state)
        state['final_answer'] = final_answer
        state['reasoning_trace'].append(f"Final synthesis: Generated answer")
        return state

    def route_decision(self, state: OrchestratorState) -> str:
        """
        Route based on supervisor decision.

        Args:
            state: Current workflow state

        Returns:
            Next node to execute
        """
        if state['final_answer']:
            return "end"

        if state['current_step'] >= state['max_steps']:
            return "synthesize"

        # Get last decision
        if state['history']:
            return state['history'][-1]['decision']

        return "web_agent"  # Default start

    def _synthesize_answer(self, state: OrchestratorState) -> str:
        """
        Synthesize final answer from all agent outputs using GAIA format.

        Args:
            state: Current workflow state

        Returns:
            Final answer string
        """
        # Compile all agent outputs
        outputs_summary = []

        for agent_name, output in state['agent_outputs'].items():
            if output.get('success'):
                outputs_summary.append(f"{agent_name}: {output.get('answer', output.get('interpretation', ''))}")
            else:
                outputs_summary.append(f"{agent_name}: Failed - {output.get('message', 'Unknown error')}")

        prompt = f"""Based on all the information gathered, answer this question:

Question: {state['question']}

Information gathered:
{chr(10).join(outputs_summary)}

Provide your final answer using the required format.
"""

        try:
            messages = [
                SystemMessage(content=self.gaia_prompt),  # Use GAIA system prompt
                HumanMessage(content=prompt)
            ]

            response = self.llm.invoke(messages)
            answer = response.content if hasattr(response, 'content') else str(response)

            logger.info(f"Synthesized answer: {answer[:200]}...")
            return answer

        except Exception as e:
            logger.error(f"Error synthesizing answer: {e}")
            return "Error: Unable to synthesize answer"

    def _format_history(self, state: OrchestratorState) -> str:
        """Format history for context."""
        if not state['history']:
            return "No previous actions"

        return '\n'.join([
            f"Step {h['step']}: {h['decision']}"
            for h in state['history']
        ])

    def _format_agent_outputs(self, outputs: Dict) -> str:
        """Format agent outputs for display."""
        if not outputs:
            return "No outputs yet"

        formatted = []
        for agent, output in outputs.items():
            if output.get('success'):
                formatted.append(f"- {agent}: Success")
            else:
                formatted.append(f"- {agent}: {output.get('message', 'Failed')}")

        return '\n'.join(formatted)

    def run(self, question: str, file_path: str = None) -> Dict[str, Any]:
        """
        Run orchestrator to answer question.

        Args:
            question: Question to answer
            file_path: Optional file path

        Returns:
            Dictionary with answer and reasoning trace
        """
        logger.info(f"Starting orchestrator for question: {question[:100]}...")

        # Initialize state
        initial_state: OrchestratorState = {
            'question': question,
            'file_path': file_path or '',
            'current_step': 0,
            'max_steps': self.max_iterations,
            'agent_outputs': {},
            'final_answer': '',
            'history': [],
            'reasoning_trace': []
        }

        # Run graph
        try:
            final_state = self.graph.invoke(initial_state)

            logger.info(f"Orchestrator completed in {final_state['current_step']} steps")

            return {
                'answer': final_state['final_answer'],
                'reasoning_trace': final_state['reasoning_trace'],
                'steps': final_state['current_step'],
                'agent_outputs': final_state['agent_outputs']
            }

        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            return {
                'answer': f"Error: {str(e)}",
                'reasoning_trace': [f"Error occurred: {str(e)}"],
                'steps': 0,
                'agent_outputs': {}
            }
