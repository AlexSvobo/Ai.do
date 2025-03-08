"""
AI.do Assistant
MIT License
Copyright (c) 2025 Alex Svoboda

A bridge between Stata and Large Language Models
"""

import os
import json
import logging
from typing import Dict, Optional, List, Any, TypedDict
from dataclasses import dataclass
from abc import ABC, abstractmethod
import platform
import urllib.request
import urllib.error
import urllib.parse
import json

def get_stata_path() -> str:
    """Get Stata utilities path based on OS and common install locations"""
    system = platform.system()
    
    if system == "Windows":
        # Check common Windows Stata paths in order of version
        stata_paths = [
            r"C:\Program Files\Stata18\utilities",
            r"C:\Program Files\Stata17\utilities",
            r"C:\Program Files\Stata16\utilities"
        ]
    elif system == "Darwin":  # macOS
        stata_paths = [
            "/Applications/Stata/utilities",
            "/Applications/Stata18/utilities",
            "/Applications/Stata17/utilities",
            "/Applications/Stata16/utilities"
        ]
    else:  # Linux
        stata_paths = [
            "/usr/local/stata18/utilities",
            "/usr/local/stata17/utilities",
            "/usr/local/stata16/utilities"
        ]
    
    # Return first valid path
    for path in stata_paths:
        if os.path.exists(path):
            return path
            
    raise RuntimeError("Could not find Stata utilities folder. Please check your Stata installation.")

# Import PyStata API using dynamic path
import sys
sys.path.insert(0, get_stata_path())
from pystata import config
config.init('mp')  # Initialize PyStata in memory-plus mode

# ---------------------------
# Constants & Enums
# ---------------------------

MAX_HISTORY_ITEMS = 5

# ---------------------------
# Model Provider Interfaces
# ---------------------------

class ModelProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, context: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        pass

class HuggingFaceProvider(ModelProvider):
    def __init__(self, api_key: str, model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.api_key = api_key
        self.model_id = model_id
        # Update endpoints and format settings
        self.inference_endpoints = {
            "together": {
                "url": "https://router.huggingface.co/together/v1/chat/completions",
                "is_chat": True,
                "max_tokens": 2048  # Increased token limit
            },
            "huggingface": {
                "url": f"https://api-inference.huggingface.co/models/{model_id}",
                "is_chat": False,
                "max_tokens": 1024  # Standard limit for inference API
            }
        }

    def validate_connection(self) -> bool:
        """Validate the API connection by trying a simple query"""
        try:
            test_response = self.generate_response("Test connection", {})
            return bool(test_response)
        except Exception as e:
            logger.warning(f"Connection validation failed: {e}")
            return False
        
    def generate_response(self, prompt: str, context: Dict[str, Any]) -> str:
        for provider, config in self.inference_endpoints.items():
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                # Adjust payload based on endpoint type
                if config["is_chat"]:
                    payload = {
                        "model": self.model_id,
                        "messages": [{
                            "role": "user",
                            "content": prompt
                        }],
                        "max_tokens": config["max_tokens"],
                        "stream": False,
                        "temperature": 0.7,
                        "stop": ["<|endoftext|>", "Human:", "Assistant:"]  # Stop tokens
                    }
                else:
                    payload = {
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": config["max_tokens"],
                            "temperature": 0.7,
                            "return_full_text": False
                        }
                    }
                
                # Use urllib from standard library
                req = urllib.request.Request(
                    config["url"],
                    data=json.dumps(payload).encode(),
                    headers=headers
                )
                
                with urllib.request.urlopen(req) as response:
                    result = json.loads(response.read().decode())
                    
                    # Extract response text based on format
                    response_text = ""
                    if config["is_chat"]:
                        if "choices" in result:
                            response_text = result["choices"][0]["message"]["content"]
                    else:
                        if isinstance(result, list) and result:
                            response_text = result[0].get("generated_text", "")
                    
                    # Clean up and format response
                    if response_text:
                        # Remove common artifacts
                        response_text = response_text.replace("<|endoftext|>", "")
                        response_text = response_text.strip()
                        
                        # Format code blocks for Stata
                        lines = []
                        in_code_block = False
                        for line in response_text.split("\n"):
                            if "```stata" in line:
                                in_code_block = True
                                lines.append(". // Code block start")
                            elif "```" in line and in_code_block:
                                in_code_block = False
                                lines.append(". // Code block end")
                            else:
                                lines.append(line)
                        
                        return "\n".join(lines)
                    
            except Exception as e:
                logger.warning(f"Failed with {provider} endpoint: {e}")
                continue
                
        raise Exception("All inference endpoints failed")

class GeminiProvider(ModelProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent"
        
    def generate_response(self, prompt: str, context: Dict[str, Any]) -> str:
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        url = f"{self.api_url}?key={self.api_key}"
        
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers=headers
            )
            
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                
                if "error" in result:
                    raise Exception(f"API error: {result['error']}")
                
                return result["candidates"][0]["content"]["parts"][0]["text"]
                
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def validate_connection(self) -> bool:
        try:
            # Simple test request
            test_response = self.generate_response("Test connection", {})
            return bool(test_response)
        except Exception:
            return False

# ---------------------------
# Context Management
# ---------------------------

@dataclass
class DatasetContext:
    variables: List[str]
    types: List[str]
    observations: int
    has_time_var: bool = False
    has_panel_id: bool = False
    metadata: Dict[str, Any] = None
    
    @classmethod
    def from_stata_describe(cls, describe_output: str) -> 'DatasetContext':
        """Parse Stata describe output properly"""
        try:
            lines = describe_output.split('\n')
            variables = []
            types = []
            obs_count = 0
            
            for line in lines:
                if "obs:" in line:
                    obs_count = int(line.split("obs:")[1].strip().split()[0])
                elif line.strip() and not line.startswith("storage"):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        variables.append(parts[0])
                        types.append(parts[1])
            
            return cls(
                variables=variables,
                types=types,
                observations=obs_count,
                metadata={"source": describe_output}
            )
        except Exception as e:
            logging.error(f"Error parsing Stata output: {e}")
            raise

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variables": self.variables,
            "types": self.types,
            "observations": self.observations,
            "has_time_var": self.has_time_var,
            "has_panel_id": self.has_panel_id,
            "metadata": self.metadata
        }

class VariableInfo(TypedDict):
    type: str
    label: str
    statistics: Dict[str, Optional[float]]
    value_labels: Dict[str, str]

from sfi import Macro, Scalar, Matrix
from typing import Dict, List, Optional
import math

class EstimationResults:
    """Helper class for Stata results (now simplified since most work is done in Stata)"""
    
    @staticmethod
    def get_current_results() -> str:
        """This is now just a fallback in case there are no results from Stata"""
        from sfi import Macro
        
        # We now primarily rely on the results captured in Stata
        # This method only serves as a fallback
        try:
            cmd = Macro.getGlobal("e(cmd)")
            if cmd:
                return f"Current estimation command: {cmd}"
            return "No current estimation results found."
        except:
            return "Unable to retrieve estimation results."

class ContextManager:
    def __init__(self):
        self.dataset_context: Optional[DatasetContext] = None
        self.command_history: List[str] = []
        self.session_state: Dict[str, Any] = {}
        self.estimation_results: Optional[Dict[str, Any]] = None
        self.chat_history: List[Dict[str, str]] = []  # Add chat history storage
        
    def update_context(self, stata_output: str):
        self.dataset_context = DatasetContext.from_stata_describe(stata_output)
    
    def add_command(self, command: str):
        self.command_history.append(command)
        if len(self.command_history) > MAX_HISTORY_ITEMS:
            self.command_history.pop(0)
        
    def get_context_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset_context.to_dict() if self.dataset_context else {},
            "history": self.command_history,
            "state": self.session_state,
            "estimation": self.estimation_results if self.estimation_results else {},
            "chat_history": self.chat_history  # Add chat history to context
        }

    def clear_history(self):
        self.command_history.clear()

    def update_estimation_results(self, results: Dict[str, Any]):
        """Update estimation results in context"""
        self.estimation_results = results

    def add_chat_message(self, role: str, content: str):
        """Add a chat message to the history"""
        self.chat_history.append({
            "role": role,
            "content": content
        })
        # Keep last 10 messages to avoid context getting too large
        if len(self.chat_history) > 10:
            self.chat_history.pop(0)

    def update_command_history(self):
        """Update command history using direct command execution"""
        try:
            from sfi import Data
            # Execute the history command and capture output
            history_output = Data.execCommand("history", True)
            
            if history_output:
                commands = []
                lines = history_output.split("\n")
                for line in lines:
                    # History output typically shows line numbers like "  1. command"
                    if "." in line:
                        parts = line.split(".", 1)
                        if len(parts) > 1:
                            cmd = parts[1].strip()
                            if cmd:
                                commands.append(cmd.strip())
                
                # Update history with parsed commands (most recent first)
                self.command_history = commands[:MAX_HISTORY_ITEMS]
            
        except Exception as e:
            # Fallback to existing method if this fails
            try:
                from sfi import Macro
                history = []
                for i in range(20):  # Try up to 20 recent commands
                    try:
                        cmd = Macro.getGlobal(f"c(cmdline{i})")
                        if cmd and cmd.strip():
                            history.append(cmd.strip())
                    except:
                        continue
                
                self.command_history = history[:MAX_HISTORY_ITEMS]
            except:
                # Keep existing history if all else fails
                pass

# Core AI.do Assistant Class
class AiDoAssistant:
    def __init__(self, provider: ModelProvider):
        self.provider = provider
        self.context_manager = ContextManager()
        
    def process_query(self, query: str) -> str:
        """Process query using comprehensive Stata results"""
        # Add user query to chat history
        self.context_manager.add_chat_message("user", query)
        
        # Update command history from Stata
        self.context_manager.update_command_history()
        
        # Get base context with chat history
        context = self.context_manager.get_context_dict()
        formatted_variables = self._format_variables(context)
        
        # Use the comprehensive results directly from session state
        results_text = self.context_manager.session_state.get("stata_results_raw", "")
        
        # Add results to context
        context["recent_results"] = results_text
        
        formatted_prompt = self._get_prompt_template().format(
            query=query,
            context=context,
            formatted_variables=formatted_variables
        )
        
        # Get AI response
        response = self.provider.generate_response(formatted_prompt, context)
        
        # Add AI response to chat history
        self.context_manager.add_chat_message("assistant", response)
        
        return response

    def _format_variables(self, context: Dict[str, Any]) -> str:
        """Optimized variable formatting with list comprehension and missing value handling"""
        metadata = context.get("dataset", {}).get("metadata", {}).get("columns", {})
        
        variables = [
            f"- {name} ({info['type']})"
            + (f": {info['label']}" if info['label'] else "")
            + (f" [Statistics: {', '.join(f'{k}={v:.2f}' for k, v in info['statistics'].items() if v is not None)}]" if info['statistics'] else "")
            + (f" [Labels: {', '.join(f'{k}={v}' for k, v in info['value_labels'].items())}]" if info['value_labels'] else "")
            + (f" [Missing: {info['statistics'].get('missing', 0)} observations]" if info['statistics'].get('missing', 0) > 0 else "")
            for name, info in metadata.items()
        ]

        # Count variables with value labels and missing values
        vars_with_value_labels = sum(1 for info in metadata.values() if info['value_labels'])
        vars_with_missing = sum(1 for info in metadata.values() if info['statistics'].get('missing', 0) > 0)
        total_missing = sum(info['statistics'].get('missing', 0) for info in metadata.values())
        
        # Calculate statistics for summary
        total_vars = len(metadata)
        numeric_vars = sum(1 for info in metadata.values() if info['type'] not in ['str', 'strL'])
        string_vars = sum(1 for info in metadata.values() if info['type'].startswith('str'))
        vars_with_labels = sum(1 for info in metadata.values() if info['label'])
        
        summary = f"""
        Dataset Summary:
        - Total Variables: {total_vars}
        - Numeric Variables: {numeric_vars}
        - String Variables: {string_vars}
        - Variables with Labels: {vars_with_labels}
        - Variables with Value Labels: {vars_with_value_labels}
        - Variables with Missing Values: {vars_with_missing}
        - Total Missing Values: {total_missing}
        """
        
        # Add explicit notes about value labels and missing values
        if vars_with_value_labels == 0:
            summary += "\n    Note: No value labels are currently defined. This means that numeric variables represent their values directly."
        
        # Get missing value patterns from sample data
        missing_patterns = context.get("dataset", {}).get("metadata", {}).get("sample_data", {}).get("missing_patterns", {})
        if missing_patterns:
            summary += "\n    Missing Value Information:"
            for var, patterns in missing_patterns.items():
                pattern_str = ", ".join(f"{code}({count})" for code, count in patterns.items())
                summary += f"\n    - {var}: {pattern_str}"
        
        # Add Stata-specific missing value explanation
        if vars_with_missing > 0:
            summary += "\n\n    Note about Stata missing values:"
            summary += "\n    - '.' represents standard missing value"
            summary += "\n    - '.a' through '.z' represent extended missing values"
            summary += "\n    - Missing values are ordered: . < .a < .b < ... < .z"
        
        # Add sample data information if available
        sample_data = context.get("dataset", {}).get("metadata", {}).get("sample_data", {}).get("rows", [])
        if sample_data:
            summary += f"\n\n    Sample Data: {len(sample_data)} rows available for context"
            
            # Include first row as an example
            if sample_data[0]:
                first_row = {k: v for k, v in list(sample_data[0].items())[:5]}  # Limit to first 5 columns
                summary += f"\n    Example (first row): {json.dumps(first_row, default=str)}"
                if len(sample_data[0]) > 5:
                    summary += " ..."
        
        return "\n".join(variables) + "\n" + summary

    @staticmethod
    def _get_prompt_template() -> str:
        """Updated template to include chat history"""
        return """
        Act as a Stata expert.
        Answer the following query using the provided dataset, estimation context, and chat history:
        
        Previous Messages:
        {context[chat_history]}
        
        User Query: {query}
        
        Dataset Information:
        - Total Observations: {context[dataset][observations]}
        - Variables:
        {formatted_variables}
        
        Stata Results:
        {context[recent_results]}
        
        Command History:
        {context[history]}
        
        Provide a detailed response that directly addresses the user's query. 
        Include relevant Stata code examples when appropriate.
        """
    
    def update_dataset_context(self, stata_output: str):
        self.context_manager.update_context(stata_output)

# Configuration Management
logger = logging.getLogger(__name__)

class Config:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.provider = None
        self.api_key = None
        self.model_id = None
        self._load_config()

    def _load_config(self):
        """Load config from simple text file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    if len(lines) >= 2:
                        self.provider = lines[0]
                        self.api_key = lines[1]
                        self.model_id = lines[2] if len(lines) > 2 else None
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def get_provider_config(self, provider_name: str) -> dict:
        """Get provider settings as simple dict"""
        if provider_name != self.provider:
            raise ValueError(f"Provider mismatch: {provider_name} vs {self.provider}")
        return {
            "api_key": self.api_key,
            "model_id": self.model_id
        }

# Factory for creating provider instances
class ProviderFactory:
    @staticmethod
    def create_provider(provider_name: str, config: Config) -> ModelProvider:
        provider_config = config.get_provider_config(provider_name)
        
        if provider_name == "huggingface":
            return HuggingFaceProvider(
                api_key=provider_config["api_key"],
                model_id=provider_config.get("model_id")
            )
        elif provider_name == "gemini":
            return GeminiProvider(api_key=provider_config["api_key"])
        else:
            raise ValueError(f"Unknown provider: {provider_name}")