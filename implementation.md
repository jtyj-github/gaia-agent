# GAIA Multi-Agent System Implementation Guide

## Project Goal
Build a LangGraph-based multi-agent system to achieve **30% accuracy on the GAIA benchmark** for the HuggingFace AI Agents course final project.

## Prerequisites
- Python 3.10+
- 16GB+ RAM (for Ollama)
- 8GB+ VRAM (recommended for local inference)
- Docker installed (for code sandboxing)
- Git

---

## Phase 1: Environment Setup (Day 1)

### Step 1.1: Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### Step 1.2: Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium
```

### Step 1.3: Set Up Ollama (Local Inference)
```bash
# Download and install Ollama from https://ollama.ai

# Pull recommended models
ollama pull llama3.1:8b
ollama pull mistral:7b

# Test Ollama
ollama run llama3.1:8b "Hello, world!"
```

### Step 1.4: Set Up Environment Variables
Create a `.env` file in the root directory:
```bash
# .env file
# Model Configuration
MODEL_PROVIDER=ollama  # Options: ollama, huggingface
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434

# HuggingFace Fallback
HF_TOKEN=your_huggingface_token_here
HF_MODEL=meta-llama/Llama-3.1-8B-Instruct

# API Keys (get these as needed)
OPENAI_API_KEY=your_openai_key_here  # For vision API
WOLFRAM_ALPHA_APP_ID=your_wolfram_key_here
GOOGLE_API_KEY=your_google_key_here
GOOGLE_CSE_ID=your_google_cse_id_here

# Docker Settings
DOCKER_TIMEOUT=30

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/gaia_agent.log
```

### Step 1.5: Test Hardware Capabilities
Create `test_hardware.py`:
```python
import psutil
import subprocess

print(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
print(f"CPU Cores: {psutil.cpu_count()}")

# Test GPU
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print("GPU Available:", "NVIDIA" in result.stdout)
except:
    print("No NVIDIA GPU detected")

# Test Ollama
try:
    import ollama
    models = ollama.list()
    print(f"Ollama models: {[m['name'] for m in models['models']]}")
except:
    print("Ollama not running or not installed")
```

Run: `python test_hardware.py`

**Decision Point:** If hardware is insufficient (<16GB RAM or Ollama too slow), set `MODEL_PROVIDER=huggingface` in `.env`

---

## Phase 2: Configuration Files (Day 1-2)

### Step 2.1: Model Configuration (`config/model_config.yaml`)
```yaml
# Model settings
model:
  provider: ollama  # ollama or huggingface

  ollama:
    base_url: http://localhost:11434
    model: llama3.1:8b
    temperature: 0.1
    timeout: 60

  huggingface:
    model: meta-llama/Llama-3.1-8B-Instruct
    temperature: 0.1
    max_tokens: 2048

  # Model selection by task
  routing:
    orchestrator: llama3.1:8b  # Main coordinator
    web_agent: llama3.1:8b
    code_executor: llama3.1:8b
    file_handler: llama3.1:8b
    validator: llama3.1:8b
```

### Step 2.2: Agent Configuration (`config/agent_config.yaml`)
```yaml
# Agent settings
orchestrator:
  max_iterations: 10
  timeout: 300  # 5 minutes per question
  system_prompt: |
    You are an orchestrator agent coordinating specialized agents to answer complex questions.
    Available agents:
    - web_agent: Web search and browsing
    - code_executor: Execute Python/bash code
    - file_handler: Process files (PDF, Excel, images, audio)
    - validator: Validate and format final answers

    Your job:
    1. Analyze the question
    2. Delegate to appropriate agents
    3. Synthesize information
    4. Return a validated answer

web_agent:
  max_search_results: 5
  max_pages_to_visit: 3
  timeout: 30
  system_prompt: |
    You are a web research agent. You can search the web and extract information from pages.
    Be thorough but efficient. Focus on authoritative sources.

code_executor:
  enable_python: true
  enable_bash: true
  timeout: 30
  max_output_length: 10000
  system_prompt: |
    You are a code execution agent. Execute Python or bash code safely in a sandbox.
    Return clear, structured output.

file_handler:
  supported_formats:
    - pdf
    - xlsx
    - xls
    - csv
    - docx
    - png
    - jpg
    - mp3
    - mp4
  system_prompt: |
    You are a file processing agent. Extract and analyze information from various file types.

validator:
  normalization_rules:
    - lowercase
    - strip_whitespace
    - remove_punctuation
  system_prompt: |
    You are a validation agent. Ensure answers are:
    1. Accurate and directly answer the question
    2. In the correct format (number, date, yes/no, etc.)
    3. Normalized for comparison
```

### Step 2.3: Tool Configuration (`config/tool_config.yaml`)
```yaml
# Tool settings
search:
  provider: duckduckgo  # Free, no API key needed
  max_results: 5
  region: us-en
  safe_search: moderate

browser:
  engine: playwright  # or selenium
  headless: true
  timeout: 30
  viewport:
    width: 1280
    height: 720

code_execution:
  sandbox: docker
  timeout: 30
  memory_limit: 512m
  network_enabled: false

file_processing:
  pdf:
    engine: pdfplumber  # or PyPDF2
    extract_images: true
  excel:
    engine: openpyxl
    read_only: true
  image:
    ocr_enabled: true
    vision_api: openai  # Requires API key
  audio:
    transcription: whisper
    model: base

math:
  symbolic_engine: sympy
  wolfram_enabled: true

caching:
  enabled: true
  ttl: 3600  # 1 hour
  max_size: 1000
```

---

## Phase 3: Utilities Infrastructure (Day 2)

### Step 3.1: Logger Setup (`utils/logger.py`)
```python
import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """Set up logger with file and console handlers."""

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # File handler
    log_file = log_dir / f"{name}_{datetime.now():%Y%m%d}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
```

### Step 3.2: Cache Implementation (`utils/cache.py`)
```python
from typing import Any, Optional
from datetime import datetime, timedelta
import json
import hashlib
from pathlib import Path

class SimpleCache:
    """Simple file-based cache for tool results."""

    def __init__(self, cache_dir: str = ".cache", ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = ttl

    def _get_key(self, key: str) -> str:
        """Generate cache key hash."""
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached value if not expired."""
        cache_file = self.cache_dir / f"{self._get_key(key)}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)

            # Check expiration
            cached_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - cached_time > timedelta(seconds=self.ttl):
                cache_file.unlink()  # Delete expired cache
                return None

            return data['value']
        except:
            return None

    def set(self, key: str, value: Any):
        """Cache a value."""
        cache_file = self.cache_dir / f"{self._get_key(key)}.json"

        data = {
            'timestamp': datetime.now().isoformat(),
            'value': value
        }

        with open(cache_file, 'w') as f:
            json.dump(data, f)

    def clear(self):
        """Clear all cache."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
```

### Step 3.3: Utils Init (`utils/__init__.py`)
```python
from .logger import setup_logger
from .cache import SimpleCache

__all__ = ['setup_logger', 'SimpleCache']
```

---

## Phase 4: Core Tools Implementation (Day 3-4)

### Step 4.1: Search Tools (`tools/search_tools.py`)
```python
from duckduckgo_search import DDGS
from typing import List, Dict
from utils import setup_logger, SimpleCache

logger = setup_logger("search_tools")
cache = SimpleCache()

def web_search(query: str, max_results: int = 5) -> List[Dict]:
    """Search the web using DuckDuckGo."""

    # Check cache
    cache_key = f"search:{query}:{max_results}"
    cached = cache.get(cache_key)
    if cached:
        logger.info(f"Cache hit for query: {query}")
        return cached

    logger.info(f"Searching: {query}")

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        # Cache results
        cache.set(cache_key, results)
        return results

    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

def search_news(query: str, max_results: int = 5) -> List[Dict]:
    """Search news articles."""
    try:
        with DDGS() as ddgs:
            return list(ddgs.news(query, max_results=max_results))
    except Exception as e:
        logger.error(f"News search error: {e}")
        return []
```

### Step 4.2: Browser Tools (`tools/browser_tools.py`)
```python
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from typing import Optional
from utils import setup_logger

logger = setup_logger("browser_tools")

def fetch_page_content(url: str, timeout: int = 30) -> Optional[str]:
    """Fetch page content using Playwright."""
    logger.info(f"Fetching: {url}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=timeout * 1000)

            # Wait for content to load
            page.wait_for_load_state("networkidle")

            content = page.content()
            browser.close()

            return content

    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return None

def extract_text_from_html(html: str) -> str:
    """Extract clean text from HTML."""
    soup = BeautifulSoup(html, 'lxml')

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Get text
    text = soup.get_text()

    # Clean up
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text

def scrape_page(url: str) -> Optional[Dict]:
    """Scrape page and return structured data."""
    html = fetch_page_content(url)
    if not html:
        return None

    soup = BeautifulSoup(html, 'lxml')

    return {
        'url': url,
        'title': soup.title.string if soup.title else '',
        'text': extract_text_from_html(html),
        'links': [a.get('href') for a in soup.find_all('a', href=True)][:20]
    }
```

### Step 4.3: Code Tools (`tools/code_tools.py`)
```python
import docker
import tempfile
import os
from typing import Dict, Optional
from utils import setup_logger

logger = setup_logger("code_tools")

class CodeExecutor:
    """Execute code in Docker sandbox."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.client = docker.from_env()

    def execute_python(self, code: str) -> Dict:
        """Execute Python code in sandbox."""
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

            return {
                'success': result['StatusCode'] == 0,
                'output': output,
                'exit_code': result['StatusCode']
            }

        except Exception as e:
            logger.error(f"Python execution error: {e}")
            return {
                'success': False,
                'output': str(e),
                'exit_code': -1
            }

    def execute_bash(self, command: str) -> Dict:
        """Execute bash command in sandbox."""
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
    """Convenience function for Python execution."""
    return executor.execute_python(code)

def run_bash_command(command: str) -> Dict:
    """Convenience function for bash execution."""
    return executor.execute_bash(command)
```

### Step 4.4: File Tools (`tools/file_tools.py`)
```python
import PyPDF2
import pdfplumber
import openpyxl
import pandas as pd
from PIL import Image
import pytesseract
from pathlib import Path
from typing import Dict, Any, Optional
from utils import setup_logger

logger = setup_logger("file_tools")

def read_pdf(file_path: str, method: str = "pdfplumber") -> str:
    """Extract text from PDF."""
    logger.info(f"Reading PDF: {file_path}")

    try:
        if method == "pdfplumber":
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                return text

        else:  # PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                return text

    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        return ""

def read_excel(file_path: str) -> Dict[str, Any]:
    """Read Excel file and return structured data."""
    logger.info(f"Reading Excel: {file_path}")

    try:
        df = pd.read_excel(file_path, sheet_name=None)

        result = {}
        for sheet_name, sheet_df in df.items():
            result[sheet_name] = {
                'columns': sheet_df.columns.tolist(),
                'rows': sheet_df.values.tolist(),
                'summary': sheet_df.describe().to_dict()
            }

        return result

    except Exception as e:
        logger.error(f"Error reading Excel: {e}")
        return {}

def read_csv(file_path: str) -> Dict[str, Any]:
    """Read CSV file."""
    logger.info(f"Reading CSV: {file_path}")

    try:
        df = pd.read_csv(file_path)
        return {
            'columns': df.columns.tolist(),
            'rows': df.values.tolist(),
            'shape': df.shape,
            'summary': df.describe().to_dict()
        }
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return {}

def extract_text_from_image(image_path: str) -> str:
    """Extract text from image using OCR."""
    logger.info(f"OCR on image: {image_path}")

    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logger.error(f"Error with OCR: {e}")
        return ""

def get_image_info(image_path: str) -> Dict[str, Any]:
    """Get image metadata."""
    try:
        image = Image.open(image_path)
        return {
            'format': image.format,
            'size': image.size,
            'mode': image.mode,
            'has_text': bool(extract_text_from_image(image_path).strip())
        }
    except Exception as e:
        logger.error(f"Error reading image: {e}")
        return {}

def process_file(file_path: str) -> Dict[str, Any]:
    """Process file based on extension."""
    path = Path(file_path)
    ext = path.suffix.lower()

    handlers = {
        '.pdf': lambda: {'type': 'pdf', 'content': read_pdf(file_path)},
        '.xlsx': lambda: {'type': 'excel', 'content': read_excel(file_path)},
        '.xls': lambda: {'type': 'excel', 'content': read_excel(file_path)},
        '.csv': lambda: {'type': 'csv', 'content': read_csv(file_path)},
        '.png': lambda: {'type': 'image', 'info': get_image_info(file_path), 'text': extract_text_from_image(file_path)},
        '.jpg': lambda: {'type': 'image', 'info': get_image_info(file_path), 'text': extract_text_from_image(file_path)},
        '.jpeg': lambda: {'type': 'image', 'info': get_image_info(file_path), 'text': extract_text_from_image(file_path)},
    }

    handler = handlers.get(ext)
    if handler:
        return handler()
    else:
        logger.warning(f"Unsupported file type: {ext}")
        return {'type': 'unknown', 'error': f'Unsupported file type: {ext}'}
```

### Step 4.5: Math Tools (`tools/math_tools.py`)
```python
import sympy
import numpy as np
from typing import Any, Optional
from utils import setup_logger

logger = setup_logger("math_tools")

def calculate(expression: str) -> Optional[float]:
    """Evaluate mathematical expression."""
    try:
        result = sympy.sympify(expression)
        return float(result.evalf())
    except Exception as e:
        logger.error(f"Calculation error: {e}")
        return None

def solve_equation(equation: str, variable: str = 'x') -> list:
    """Solve equation symbolically."""
    try:
        x = sympy.Symbol(variable)
        eq = sympy.sympify(equation)
        solutions = sympy.solve(eq, x)
        return [float(sol.evalf()) for sol in solutions]
    except Exception as e:
        logger.error(f"Equation solving error: {e}")
        return []

def differentiate(expression: str, variable: str = 'x') -> str:
    """Compute derivative."""
    try:
        x = sympy.Symbol(variable)
        expr = sympy.sympify(expression)
        derivative = sympy.diff(expr, x)
        return str(derivative)
    except Exception as e:
        logger.error(f"Differentiation error: {e}")
        return ""

def integrate(expression: str, variable: str = 'x') -> str:
    """Compute integral."""
    try:
        x = sympy.Symbol(variable)
        expr = sympy.sympify(expression)
        integral = sympy.integrate(expr, x)
        return str(integral)
    except Exception as e:
        logger.error(f"Integration error: {e}")
        return ""

def matrix_operations(operation: str, matrix_a: list, matrix_b: list = None) -> Any:
    """Perform matrix operations."""
    try:
        A = np.array(matrix_a)

        if operation == "determinant":
            return float(np.linalg.det(A))
        elif operation == "inverse":
            return np.linalg.inv(A).tolist()
        elif operation == "eigenvalues":
            return np.linalg.eigvals(A).tolist()
        elif operation == "multiply" and matrix_b:
            B = np.array(matrix_b)
            return np.matmul(A, B).tolist()
        else:
            return None
    except Exception as e:
        logger.error(f"Matrix operation error: {e}")
        return None
```

---

## Phase 5: Agent Implementations (Day 5-7)

### Step 5.1: Validator Agent (`agents/validator.py`)
```python
import re
from typing import Any, Dict
from utils import setup_logger

logger = setup_logger("validator")

class AnswerValidator:
    """Validate and normalize answers for GAIA evaluation."""

    def normalize(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if not answer:
            return ""

        # Convert to string
        answer = str(answer).strip()

        # Lowercase
        answer = answer.lower()

        # Remove extra whitespace
        answer = re.sub(r'\s+', ' ', answer)

        # Remove common punctuation
        answer = re.sub(r'[.,;:!?]$', '', answer)

        return answer

    def validate_format(self, answer: str, expected_format: str) -> bool:
        """Check if answer matches expected format."""

        if expected_format == "number":
            return bool(re.match(r'^-?\d+(\.\d+)?$', answer.strip()))

        elif expected_format == "date":
            # Check common date formats
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
            ]
            return any(re.match(pattern, answer.strip()) for pattern in date_patterns)

        elif expected_format == "yes_no":
            return answer.strip().lower() in ['yes', 'no', 'true', 'false']

        return True  # Default: accept any format

    def extract_final_answer(self, text: str) -> str:
        """Extract final answer from agent response."""

        # Look for common patterns
        patterns = [
            r'final answer[:\s]+(.+)',
            r'answer[:\s]+(.+)',
            r'the answer is[:\s]+(.+)',
            r'result[:\s]+(.+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # If no pattern found, return last line
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        return lines[-1] if lines else text

    def validate(self, answer: Any, expected_format: str = None) -> Dict[str, Any]:
        """Validate answer and return structured result."""

        answer_str = str(answer).strip()

        result = {
            'original': answer_str,
            'normalized': self.normalize(answer_str),
            'valid': True,
            'format_match': True
        }

        if expected_format:
            result['format_match'] = self.validate_format(answer_str, expected_format)

        return result

# Global validator instance
validator = AnswerValidator()
```

### Step 5.2: Web Agent (`agents/web_agent.py`)
```python
from typing import List, Dict, Optional
from langchain.schema import SystemMessage, HumanMessage
from tools.search_tools import web_search
from tools.browser_tools import scrape_page
from utils import setup_logger

logger = setup_logger("web_agent")

class WebAgent:
    """Agent for web search and browsing."""

    def __init__(self, llm, config: Dict):
        self.llm = llm
        self.config = config
        self.max_search_results = config.get('max_search_results', 5)
        self.max_pages = config.get('max_pages_to_visit', 3)

    def search_and_extract(self, query: str) -> Dict[str, Any]:
        """Search web and extract relevant information."""
        logger.info(f"Web agent processing query: {query}")

        # Step 1: Search
        search_results = web_search(query, max_results=self.max_search_results)

        if not search_results:
            return {'success': False, 'message': 'No search results found'}

        # Step 2: Visit top pages
        scraped_data = []
        for i, result in enumerate(search_results[:self.max_pages]):
            url = result.get('href') or result.get('link')
            if url:
                page_data = scrape_page(url)
                if page_data:
                    scraped_data.append(page_data)

        # Step 3: LLM summarization
        context = self._format_context(search_results, scraped_data)

        prompt = f"""Based on the following web search results and page contents, answer the question: {query}

Search Results:
{context}

Provide a clear, concise answer based on the information found. If the information is not sufficient, state what additional information is needed.
"""

        messages = [
            SystemMessage(content=self.config.get('system_prompt', '')),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(messages)

        return {
            'success': True,
            'answer': response.content,
            'sources': [r.get('href') or r.get('link') for r in search_results[:3]],
            'raw_data': {
                'search_results': search_results,
                'scraped_pages': scraped_data
            }
        }

    def _format_context(self, search_results: List[Dict], scraped_data: List[Dict]) -> str:
        """Format search and scrape data for LLM."""
        context = []

        for i, result in enumerate(search_results):
            context.append(f"[{i+1}] {result.get('title', 'No title')}")
            context.append(f"    URL: {result.get('href') or result.get('link')}")
            context.append(f"    Snippet: {result.get('body', 'No description')}")
            context.append("")

        if scraped_data:
            context.append("\n--- Page Contents ---\n")
            for i, page in enumerate(scraped_data):
                context.append(f"Page {i+1}: {page['title']}")
                context.append(f"{page['text'][:2000]}...")  # Limit text
                context.append("")

        return '\n'.join(context)
```

### Step 5.3: Code Executor Agent (`agents/code_executor.py`)
```python
from typing import Dict, Any
from langchain.schema import SystemMessage, HumanMessage
from tools.code_tools import run_python_code, run_bash_command
from utils import setup_logger

logger = setup_logger("code_executor")

class CodeExecutorAgent:
    """Agent for executing code to solve problems."""

    def __init__(self, llm, config: Dict):
        self.llm = llm
        self.config = config
        self.enable_python = config.get('enable_python', True)
        self.enable_bash = config.get('enable_bash', True)

    def execute(self, task: str, language: str = "python") -> Dict[str, Any]:
        """Execute code to solve a task."""
        logger.info(f"Code executor processing task: {task[:100]}...")

        # Step 1: Generate code using LLM
        code = self._generate_code(task, language)

        if not code:
            return {'success': False, 'message': 'Failed to generate code'}

        # Step 2: Execute code
        if language.lower() == "python" and self.enable_python:
            result = run_python_code(code)
        elif language.lower() == "bash" and self.enable_bash:
            result = run_bash_command(code)
        else:
            return {'success': False, 'message': f'Language {language} not enabled'}

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
        """Generate code using LLM."""

        prompt = f"""Write {language} code to solve the following task:

Task: {task}

Requirements:
- Write only the code, no explanations
- Make it self-contained
- Print the final result
- Handle errors gracefully

Code:
"""

        messages = [
            SystemMessage(content=self.config.get('system_prompt', '')),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(messages)

        # Extract code from response
        code = response.content.strip()

        # Remove markdown code blocks if present
        if code.startswith('```'):
            lines = code.split('\n')
            code = '\n'.join(lines[1:-1])

        return code

    def _interpret_results(self, task: str, code: str, result: Dict) -> str:
        """Interpret code execution results."""

        if not result['success']:
            return f"Code execution failed: {result['output']}"

        prompt = f"""Given the task and code execution output, provide a clear answer:

Task: {task}

Code executed:
{code}

Output:
{result['output']}

What is the final answer to the task?
"""

        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)

        return response.content
```

### Step 5.4: File Handler Agent (`agents/file_handler.py`)
```python
from typing import Dict, Any
from pathlib import Path
from langchain.schema import SystemMessage, HumanMessage
from tools.file_tools import process_file
from utils import setup_logger

logger = setup_logger("file_handler")

class FileHandlerAgent:
    """Agent for processing various file types."""

    def __init__(self, llm, config: Dict):
        self.llm = llm
        self.config = config
        self.supported_formats = config.get('supported_formats', [])

    def process(self, file_path: str, question: str) -> Dict[str, Any]:
        """Process file and answer question about it."""
        logger.info(f"Processing file: {file_path}")

        # Check if file exists
        if not Path(file_path).exists():
            return {'success': False, 'message': f'File not found: {file_path}'}

        # Check if format is supported
        ext = Path(file_path).suffix.lower()[1:]
        if ext not in self.supported_formats:
            return {'success': False, 'message': f'Unsupported format: {ext}'}

        # Process file
        file_data = process_file(file_path)

        if 'error' in file_data:
            return {'success': False, 'message': file_data['error']}

        # Use LLM to answer question about file
        answer = self._answer_from_file_data(file_data, question)

        return {
            'success': True,
            'answer': answer,
            'file_type': file_data.get('type'),
            'file_data': file_data
        }

    def _answer_from_file_data(self, file_data: Dict, question: str) -> str:
        """Use LLM to answer question based on file data."""

        # Format file data for LLM
        context = self._format_file_data(file_data)

        prompt = f"""Based on the following file content, answer the question:

File Type: {file_data.get('type')}

Content:
{context}

Question: {question}

Provide a clear, specific answer based on the file content.
"""

        messages = [
            SystemMessage(content=self.config.get('system_prompt', '')),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(messages)
        return response.content

    def _format_file_data(self, file_data: Dict) -> str:
        """Format file data for LLM consumption."""

        file_type = file_data.get('type')

        if file_type == 'pdf':
            return file_data.get('content', '')[:5000]  # Limit size

        elif file_type == 'excel':
            lines = []
            for sheet, data in file_data.get('content', {}).items():
                lines.append(f"Sheet: {sheet}")
                lines.append(f"Columns: {data.get('columns')}")
                lines.append(f"First 10 rows: {data.get('rows')[:10]}")
            return '\n'.join(lines)

        elif file_type == 'csv':
            content = file_data.get('content', {})
            return f"Columns: {content.get('columns')}\nRows: {content.get('rows')[:20]}"

        elif file_type == 'image':
            info = file_data.get('info', {})
            text = file_data.get('text', '')
            return f"Image info: {info}\nExtracted text: {text}"

        return str(file_data)
```

Continue in next section due to length...

### Step 5.5: Orchestrator Agent (`agents/orchestrator.py`)
```python
from typing import Dict, Any, List
from langchain.schema import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from utils import setup_logger
import yaml

logger = setup_logger("orchestrator")

class OrchestratorState:
    """State object for LangGraph."""
    def __init__(self):
        self.question = ""
        self.file_path = None
        self.current_step = 0
        self.max_steps = 10
        self.agent_outputs = {}
        self.final_answer = None
        self.history = []

class Orchestrator:
    """Main orchestrator using LangGraph supervisor pattern."""

    def __init__(self, llm, agents: Dict, config: Dict):
        self.llm = llm
        self.agents = agents  # Dict of agent name -> agent instance
        self.config = config
        self.max_iterations = config.get('max_iterations', 10)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow."""

        workflow = StateGraph(OrchestratorState)

        # Add nodes
        workflow.add_node("supervisor", self.supervisor_node)
        workflow.add_node("web_agent", self.web_agent_node)
        workflow.add_node("code_executor", self.code_executor_node)
        workflow.add_node("file_handler", self.file_handler_node)
        workflow.add_node("validator", self.validator_node)

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
                "validator": "validator",
                "end": END
            }
        )

        # All agents return to supervisor
        for agent in ["web_agent", "code_executor", "file_handler"]:
            workflow.add_edge(agent, "supervisor")

        # Validator ends the workflow
        workflow.add_edge("validator", END)

        return workflow.compile()

    def supervisor_node(self, state: OrchestratorState) -> OrchestratorState:
        """Supervisor decides next action."""
        logger.info(f"Supervisor iteration {state.current_step}")

        # Check termination conditions
        if state.current_step >= self.max_steps:
            state.final_answer = "Max iterations reached"
            return state

        if state.final_answer:
            return state

        # Analyze question and history
        decision = self._make_decision(state)

        state.history.append({
            'step': state.current_step,
            'decision': decision
        })
        state.current_step += 1

        return state

    def _make_decision(self, state: OrchestratorState) -> str:
        """Decide which agent to use next."""

        context = self._format_history(state)

        prompt = f"""You are coordinating agents to answer a question. Analyze the question and decide the next action.

Question: {state.question}
File attached: {state.file_path is not None}

Available agents:
- web_agent: Search web and browse pages
- code_executor: Execute Python/bash code
- file_handler: Process files (PDF, Excel, images, etc.)
- validator: Validate and format final answer

Previous actions:
{context}

Agent outputs so far:
{state.agent_outputs}

Decision: Which agent should act next, or should we validate the answer?
Respond with ONLY ONE of: web_agent, code_executor, file_handler, validator, or end

Your decision:"""

        messages = [
            SystemMessage(content=self.config.get('system_prompt', '')),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(messages)
        decision = response.content.strip().lower()

        # Validate decision
        valid_decisions = ['web_agent', 'code_executor', 'file_handler', 'validator', 'end']
        if decision not in valid_decisions:
            decision = 'web_agent'  # Default

        return decision

    def web_agent_node(self, state: OrchestratorState) -> OrchestratorState:
        """Execute web agent."""
        result = self.agents['web_agent'].search_and_extract(state.question)
        state.agent_outputs['web_agent'] = result
        return state

    def code_executor_node(self, state: OrchestratorState) -> OrchestratorState:
        """Execute code executor."""
        result = self.agents['code_executor'].execute(state.question)
        state.agent_outputs['code_executor'] = result
        return state

    def file_handler_node(self, state: OrchestratorState) -> OrchestratorState:
        """Execute file handler."""
        if state.file_path:
            result = self.agents['file_handler'].process(state.file_path, state.question)
            state.agent_outputs['file_handler'] = result
        return state

    def validator_node(self, state: OrchestratorState) -> OrchestratorState:
        """Validate and format final answer."""
        # Synthesize all agent outputs
        final_answer = self._synthesize_answer(state)

        # Validate
        validated = self.agents['validator'].validate(final_answer)
        state.final_answer = validated['normalized']

        return state

    def route_decision(self, state: OrchestratorState) -> str:
        """Route based on supervisor decision."""
        if state.final_answer:
            return "end"

        if state.current_step >= self.max_steps:
            return "validator"

        # Get last decision
        if state.history:
            return state.history[-1]['decision']

        return "web_agent"  # Default start

    def _synthesize_answer(self, state: OrchestratorState) -> str:
        """Synthesize final answer from all agent outputs."""

        prompt = f"""Synthesize a final answer from the following agent outputs:

Question: {state.question}

Agent outputs:
{state.agent_outputs}

Provide a clear, concise final answer that directly addresses the question.
"""

        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)

        return response.content

    def _format_history(self, state: OrchestratorState) -> str:
        """Format history for context."""
        if not state.history:
            return "No previous actions"

        return '\n'.join([
            f"Step {h['step']}: {h['decision']}"
            for h in state.history
        ])

    def run(self, question: str, file_path: str = None) -> str:
        """Run orchestrator to answer question."""
        logger.info(f"Starting orchestrator for question: {question[:100]}...")

        # Initialize state
        state = OrchestratorState()
        state.question = question
        state.file_path = file_path

        # Run graph
        final_state = self.graph.invoke(state)

        return final_state.final_answer
```

---

## Phase 6: Main Application (Day 8)

### Step 6.1: Create Main Entry Point (`main.py`)
```python
import os
import yaml
from dotenv import load_dotenv
from pathlib import Path
from utils import setup_logger
from datasets import load_dataset

# Import agents
from agents.orchestrator import Orchestrator
from agents.web_agent import WebAgent
from agents.code_executor import CodeExecutorAgent
from agents.file_handler import FileHandlerAgent
from agents.validator import AnswerValidator

logger = setup_logger("main")

def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def initialize_llm(config: dict):
    """Initialize LLM based on configuration."""
    provider = os.getenv('MODEL_PROVIDER', config['model']['provider'])

    if provider == 'ollama':
        from langchain_community.llms import Ollama
        return Ollama(
            base_url=config['model']['ollama']['base_url'],
            model=config['model']['ollama']['model'],
            temperature=config['model']['ollama']['temperature']
        )

    elif provider == 'huggingface':
        from langchain_community.llms import HuggingFaceHub
        return HuggingFaceHub(
            repo_id=config['model']['huggingface']['model'],
            huggingfacehub_api_token=os.getenv('HF_TOKEN'),
            model_kwargs={
                'temperature': config['model']['huggingface']['temperature'],
                'max_length': config['model']['huggingface']['max_tokens']
            }
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")

def initialize_agents(llm, configs: dict) -> dict:
    """Initialize all agents."""
    agents = {
        'web_agent': WebAgent(llm, configs['web_agent']),
        'code_executor': CodeExecutorAgent(llm, configs['code_executor']),
        'file_handler': FileHandlerAgent(llm, configs['file_handler']),
        'validator': AnswerValidator()
    }
    return agents

def main():
    """Main application entry point."""

    # Load environment variables
    load_dotenv()

    logger.info("Starting GAIA Multi-Agent System")

    # Load configurations
    model_config = load_config('config/model_config.yaml')
    agent_config = load_config('config/agent_config.yaml')
    tool_config = load_config('config/tool_config.yaml')

    # Initialize LLM
    logger.info("Initializing LLM...")
    llm = initialize_llm(model_config)

    # Initialize agents
    logger.info("Initializing agents...")
    agents = initialize_agents(llm, agent_config)

    # Create orchestrator
    orchestrator = Orchestrator(llm, agents, agent_config['orchestrator'])

    # Test with a simple question
    test_question = "What is the capital of France?"
    logger.info(f"Testing with question: {test_question}")

    answer = orchestrator.run(test_question)
    logger.info(f"Answer: {answer}")

    print(f"\nQuestion: {test_question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
```

---

## Phase 7: GAIA Evaluation (Day 9)

### Step 7.1: Scorer Implementation (`evaluation/scorer.py`)
```python
from datasets import load_dataset
from typing import Dict, List
import re
from utils import setup_logger

logger = setup_logger("scorer")

class GAIAScorer:
    """Scorer for GAIA benchmark."""

    def __init__(self):
        self.dataset = None

    def load_dataset(self, split: str = "validation"):
        """Load GAIA dataset."""
        logger.info(f"Loading GAIA {split} dataset...")
        self.dataset = load_dataset("gaia-benchmark/GAIA", split=split)
        return self.dataset

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        answer = str(answer).strip().lower()
        answer = re.sub(r'\s+', ' ', answer)
        answer = re.sub(r'[.,;:!?]$', '', answer)
        return answer

    def check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth."""
        pred_norm = self.normalize_answer(predicted)
        gt_norm = self.normalize_answer(ground_truth)

        # Exact match
        if pred_norm == gt_norm:
            return True

        # Fuzzy match (contains)
        if gt_norm in pred_norm or pred_norm in gt_norm:
            return True

        # Number comparison
        try:
            pred_num = float(re.sub(r'[^\d.-]', '', predicted))
            gt_num = float(re.sub(r'[^\d.-]', '', ground_truth))
            if abs(pred_num - gt_num) < 0.01:
                return True
        except:
            pass

        return False

    def evaluate(self, predictions: Dict[str, str]) -> Dict[str, float]:
        """Evaluate predictions against ground truth."""

        if not self.dataset:
            self.load_dataset()

        correct = 0
        total = 0
        results_by_level = {1: {'correct': 0, 'total': 0},
                           2: {'correct': 0, 'total': 0},
                           3: {'correct': 0, 'total': 0}}

        for item in self.dataset:
            task_id = item['task_id']
            if task_id not in predictions:
                continue

            predicted = predictions[task_id]
            ground_truth = item['Final answer']
            level = item['Level']

            is_correct = self.check_answer(predicted, ground_truth)

            if is_correct:
                correct += 1
                results_by_level[level]['correct'] += 1

            total += 1
            results_by_level[level]['total'] += 1

            logger.info(f"Task {task_id}: {'✓' if is_correct else '✗'}")

        # Calculate metrics
        accuracy = correct / total if total > 0 else 0

        metrics = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'level_1_accuracy': results_by_level[1]['correct'] / results_by_level[1]['total'] if results_by_level[1]['total'] > 0 else 0,
            'level_2_accuracy': results_by_level[2]['correct'] / results_by_level[2]['total'] if results_by_level[2]['total'] > 0 else 0,
            'level_3_accuracy': results_by_level[3]['correct'] / results_by_level[3]['total'] if results_by_level[3]['total'] > 0 else 0,
        }

        return metrics
```

### Step 7.2: Metrics Tracking (`evaluation/metrics.py`)
```python
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from utils import setup_logger

logger = setup_logger("metrics")

class MetricsTracker:
    """Track and log performance metrics."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.current_run = {
            'start_time': datetime.now().isoformat(),
            'predictions': {},
            'metrics': {},
            'errors': []
        }

    def log_prediction(self, task_id: str, question: str, prediction: str,
                      ground_truth: str = None, correct: bool = None):
        """Log a single prediction."""
        self.current_run['predictions'][task_id] = {
            'question': question,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'correct': correct,
            'timestamp': datetime.now().isoformat()
        }

    def log_error(self, task_id: str, error: str):
        """Log an error."""
        self.current_run['errors'].append({
            'task_id': task_id,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })

    def set_metrics(self, metrics: Dict[str, Any]):
        """Set final metrics."""
        self.current_run['metrics'] = metrics
        self.current_run['end_time'] = datetime.now().isoformat()

    def save(self):
        """Save results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"results_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(self.current_run, f, indent=2)

        logger.info(f"Results saved to {output_file}")
        return output_file

    def print_summary(self):
        """Print summary of results."""
        metrics = self.current_run['metrics']

        print("\n" + "="*50)
        print("GAIA Benchmark Results")
        print("="*50)
        print(f"Overall Accuracy: {metrics.get('accuracy', 0):.2%}")
        print(f"Correct: {metrics.get('correct', 0)}/{metrics.get('total', 0)}")
        print(f"\nBy Level:")
        print(f"  Level 1: {metrics.get('level_1_accuracy', 0):.2%}")
        print(f"  Level 2: {metrics.get('level_2_accuracy', 0):.2%}")
        print(f"  Level 3: {metrics.get('level_3_accuracy', 0):.2%}")
        print(f"\nErrors: {len(self.current_run['errors'])}")
        print("="*50 + "\n")
```

### Step 7.3: Full Evaluation Script (`evaluate_gaia.py`)
```python
from main import initialize_llm, initialize_agents, load_config
from agents.orchestrator import Orchestrator
from evaluation.scorer import GAIAScorer
from evaluation.metrics import MetricsTracker
from utils import setup_logger
from dotenv import load_dotenv
from tqdm import tqdm

logger = setup_logger("evaluate")

def evaluate_on_gaia(limit: int = None):
    """Run full evaluation on GAIA benchmark."""

    # Load environment
    load_dotenv()

    # Load configs
    model_config = load_config('config/model_config.yaml')
    agent_config = load_config('config/agent_config.yaml')

    # Initialize system
    logger.info("Initializing system...")
    llm = initialize_llm(model_config)
    agents = initialize_agents(llm, agent_config)
    orchestrator = Orchestrator(llm, agents, agent_config['orchestrator'])

    # Load GAIA dataset
    scorer = GAIAScorer()
    dataset = scorer.load_dataset(split="validation")

    if limit:
        dataset = dataset.select(range(limit))

    # Initialize metrics tracker
    tracker = MetricsTracker()

    # Evaluate
    predictions = {}

    logger.info(f"Evaluating on {len(dataset)} questions...")

    for item in tqdm(dataset):
        task_id = item['task_id']
        question = item['Question']
        ground_truth = item['Final answer']
        file_path = item.get('file_path')  # If file is attached

        try:
            # Run orchestrator
            prediction = orchestrator.run(question, file_path)
            predictions[task_id] = prediction

            # Check if correct
            is_correct = scorer.check_answer(prediction, ground_truth)

            # Log
            tracker.log_prediction(task_id, question, prediction, ground_truth, is_correct)

            logger.info(f"Task {task_id}: {'✓' if is_correct else '✗'}")

        except Exception as e:
            logger.error(f"Error on task {task_id}: {e}")
            tracker.log_error(task_id, str(e))
            predictions[task_id] = ""

    # Calculate metrics
    metrics = scorer.evaluate(predictions)
    tracker.set_metrics(metrics)

    # Save results
    tracker.save()
    tracker.print_summary()

    return metrics

if __name__ == "__main__":
    # Start with 10 questions for testing
    evaluate_on_gaia(limit=10)

    # Uncomment for full evaluation
    # evaluate_on_gaia()
```

---

## Phase 8: Testing & Optimization (Day 10-14)

### Step 8.1: Unit Tests
Create `tests/test_tools.py`:
```python
import pytest
from tools.search_tools import web_search
from tools.math_tools import calculate, solve_equation
from tools.file_tools import read_pdf

def test_search():
    results = web_search("Python programming", max_results=3)
    assert len(results) > 0
    assert 'title' in results[0]

def test_calculator():
    result = calculate("2 + 2 * 3")
    assert result == 8

def test_equation_solver():
    solutions = solve_equation("x**2 - 4", "x")
    assert 2.0 in solutions or -2.0 in solutions

# Add more tests for each tool
```

### Step 8.2: Integration Tests
Create `tests/test_agents.py`:
```python
import pytest
from main import initialize_llm, initialize_agents, load_config
from agents.web_agent import WebAgent

@pytest.fixture
def setup():
    model_config = load_config('config/model_config.yaml')
    agent_config = load_config('config/agent_config.yaml')
    llm = initialize_llm(model_config)
    return llm, agent_config

def test_web_agent(setup):
    llm, config = setup
    agent = WebAgent(llm, config['web_agent'])

    result = agent.search_and_extract("What is the capital of France?")
    assert result['success']
    assert 'paris' in result['answer'].lower()

# Add more integration tests
```

### Step 8.3: Run Tests
```bash
# Install pytest
pip install pytest

# Run tests
pytest tests/ -v

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=agents --cov=tools
```

### Step 8.4: Optimization Strategies

**Prompt Engineering:**
1. Test different system prompts for each agent
2. Add few-shot examples for better routing
3. Optimize answer extraction prompts

**Tool Selection:**
1. Analyze which tools are most used
2. Remove rarely used tools
3. Add new tools based on failure analysis

**Model Selection:**
1. Test different Ollama models (Llama, Mistral, Qwen)
2. Compare with HuggingFace InferenceClient
3. Use larger models for orchestrator, smaller for agents

**Caching:**
1. Enable caching for all tools
2. Cache LLM responses for repeated queries
3. Implement result memoization

---

## Phase 9: Final Evaluation & Submission (Day 14)

### Step 9.1: Full GAIA Evaluation
```bash
# Run on full validation set
python evaluate_gaia.py

# Analyze results
python analysis/analyze_results.py  # Create this script
```

### Step 9.2: Error Analysis
Create `analysis/analyze_errors.py`:
```python
import json
from pathlib import Path
from collections import Counter

def analyze_errors(results_file):
    with open(results_file) as f:
        data = json.load(f)

    # Analyze error patterns
    errors_by_level = {1: [], 2: [], 3: []}
    errors_by_type = Counter()

    for task_id, pred in data['predictions'].items():
        if not pred['correct']:
            # Categorize error
            # Add analysis logic
            pass

    # Generate report
    print("Error Analysis Report")
    print("=" * 50)
    # Print analysis
```

### Step 9.3: Create Documentation
Update `README.md` with:
- Installation instructions
- Usage examples
- Architecture diagram
- Results and performance
- Known limitations
- Future improvements

### Step 9.4: Prepare Submission
- Clean up code
- Add docstrings
- Format with black/flake8
- Create requirements.txt
- Test on fresh environment
- Record demo video
- Submit to HuggingFace course

---

## Success Criteria

✅ **System successfully runs on GAIA validation set**
✅ **Achieves ≥30% overall accuracy**
✅ **All core tools working**
✅ **LangGraph orchestrator routing correctly**
✅ **Code is documented and tested**
✅ **Results are reproducible**

---

## Troubleshooting Guide

**Issue: Ollama too slow**
- Solution: Switch to HF InferenceClient
- Set `MODEL_PROVIDER=huggingface` in `.env`

**Issue: Docker permission errors**
- Solution: Add user to docker group
- `sudo usermod -aG docker $USER`

**Issue: Playwright not working**
- Solution: Reinstall browsers
- `playwright install chromium`

**Issue: Out of memory**
- Solution: Reduce context window
- Use smaller models
- Enable more aggressive caching

**Issue: Low accuracy**
- Optimize prompts
- Add more tools
- Improve routing logic
- Use better models

---

## Next Steps After 30%

Once you achieve 30%, consider:
1. Adding more specialized tools
2. Implementing memory/context management
3. Using GPT-4 for vision tasks
4. Adding self-reflection loops
5. Implementing better error recovery
6. Fine-tuning routing logic
7. Ensemble methods

---

Good luck with your implementation! 🚀
