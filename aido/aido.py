"""
AI.do Assistant
MIT License
Copyright (c) 2025 Alex Svoboda

A bridge between Stata and Large Language Models
"""

import os
import json
import asyncio
import logging
from typing import Dict, Optional, List, Any, TypedDict
from dataclasses import dataclass
from abc import ABC, abstractmethod
import platform

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
    async def generate_response(self, prompt: str, context: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    async def validate_connection(self) -> bool:
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

    async def validate_connection(self) -> bool:
        """Validate the API connection by trying a simple query"""
        try:
            test_response = await self.generate_response("Test connection", {})
            return bool(test_response)
        except Exception as e:
            logger.warning(f"Connection validation failed: {e}")
            return False
        
    async def generate_response(self, prompt: str, context: Dict[str, Any]) -> str:
        import aiohttp
        
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
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(config["url"], headers=headers, json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            
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
        
    async def generate_response(self, prompt: str, context: Dict[str, Any]) -> str:
        import aiohttp
        
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
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        raise Exception(f"API call failed with status {response.status}")
                    
                    result = await response.json()
                    
                    if "error" in result:
                        raise Exception(f"API error: {result['error']}")
                    
                    return result["candidates"][0]["content"]["parts"][0]["text"]
                    
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    async def validate_connection(self) -> bool:
        try:
            # TODO: Implement connection test
            return True
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
    """Helper class to get and format all available Stata estimation results"""
    
    @staticmethod
    def get_all_scalars() -> Dict[str, float]:
        """Attempts to get all e() scalars"""
        scalar_list = [
            'e(N)', 'e(df_m)', 'e(df_r)', 'e(F)', 'e(r2)', 'e(rmse)',
            'e(mss)', 'e(rss)', 'e(r2_a)', 'll', 'll_0', 'e(rank)',
            'e(N_clust)', 'e(chi2)', 'e(p)', 'e(ic)', 'e(N_g)',
            'e(g_min)', 'e(g_max)', 'e(g_avg)', 'e(sigma)', 'e(sigma_e)',
            'e(sigma_u)', 'e(r2_w)', 'e(r2_b)', 'e(r2_o)', 'e(rho)'
        ]
        results = {}
        for s in scalar_list:
            try:
                val = Scalar.getValue(s)
                if val is not None:  # Only store non-None values
                    results[s] = val
            except:
                continue
        return results

    @staticmethod
    def get_all_macros() -> Dict[str, str]:
        """Attempts to get all e() macros"""
        macro_list = [
            'e(cmd)', 'e(cmdline)', 'e(title)', 'e(marginsok)', 
            'e(vce)', 'e(depvar)', 'e(properties)', 'e(predict)',
            'e(model)', 'e(estat_cmd)', 'e(vcetype)', 'e(clustvar)',
            'e(prefix)', 'e(chi2type)', 'e(offset)', 'e(wtype)',
            'e(wexp)', 'e(title)', 'e(constraints)', 'e(predict)',
            'e(cmd)', 'e(table)', 'e(marginsok)', 'e(marginsnotok)'
        ]
        results = {}
        for m in macro_list:
            try:
                val = Macro.getGlobal(m)
                if val:  # Only store non-empty values
                    results[m] = val
            except:
                continue
        return results

    @staticmethod
    def get_coefficient_table() -> Dict[str, List[float]]:
        """Get full coefficient table with standard errors, t-stats, p-values, and CIs"""
        try:
            b_matrix = Matrix.get("e(b)")
            v_matrix = Matrix.get("e(V)")
            
            # Check if matrices exist and have content
            if b_matrix is None or v_matrix is None:
                return {}
                
            names = Matrix.getColNames("e(b)")
            if not names:  # If no column names found
                return {}
                
            coefs = b_matrix[0]
            if not coefs:  # If no coefficients found
                return {}
                
            results = {
                'names': names,
                'coef': [],
                'se': [],
                't_stat': [],
                'p_value': [],
                'ci_lower': [],
                'ci_upper': []
            }
            
            # Calculate statistics for each coefficient
            for i in range(len(coefs)):
                try:
                    coef = coefs[i]
                    if coef is None:
                        continue
                        
                    # Standard error from diagonal of variance matrix
                    se = math.sqrt(v_matrix[i][i]) if v_matrix[i][i] is not None else None
                    if se is None:
                        continue
                        
                    results['coef'].append(coef)
                    results['se'].append(se)
                    
                    # t-statistic
                    t_stat = coef / se
                    results['t_stat'].append(t_stat)
                    
                    # Simple confidence intervals (1.96 for ~95% CI)
                    results['ci_lower'].append(coef - 1.96 * se)
                    results['ci_upper'].append(coef + 1.96 * se)
                    
                    # Approximate p-value
                    try:
                        p_value = min(1, 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2)))))
                    except:
                        p_value = float('nan')
                    results['p_value'].append(p_value)
                    
                except:
                    continue
                    
            return results
        except:
            return {}

    @staticmethod
    def get_current_results() -> str:
        output = []
        
        # Get all scalars
        scalars = EstimationResults.get_all_scalars()
        if scalars:
            output.append("\n[SCALARS]")
            for name, value in scalars.items():
                if value is not None:
                    output.append(f"{name}: {value:.6g}")
        
        # Get all macros
        macros = EstimationResults.get_all_macros()
        if macros:
            output.append("\n[MACROS]")
            for name, value in macros.items():
                if value:
                    output.append(f"{name}: {value}")
        
        # Get coefficient table
        coef_table = EstimationResults.get_coefficient_table()
        if coef_table and coef_table.get('coef'):  # Check if we have coefficients
            output.append("\n[COEFFICIENT TABLE]")
            output.append("Variable | Coef. | Std.Err. | t/z | P>|t/z| | [95% Conf. Interval]")
            output.append("-" * 80)
            
            for i, name in enumerate(coef_table['names']):
                try:
                    row = (f"{name:12} | {coef_table['coef'][i]:8.4f} | "
                          f"{coef_table['se'][i]:8.4f} | {coef_table['t_stat'][i]:6.3f} | "
                          f"{coef_table['p_value'][i]:6.4f} | [{coef_table['ci_lower'][i]:8.4f}, "
                          f"{coef_table['ci_upper'][i]:8.4f}]")
                    output.append(row)
                except:
                    continue
        
        # Get matrices (just names and dimensions)
        try:
            matrix_names = ['e(b)', 'e(V)', 'e(ilog)', 'e(gradient)', 'e(S)', 'r(table)']
            matrices = []
            for m in matrix_names:
                try:
                    mat = Matrix.get(m)
                    if mat is not None and hasattr(mat, 'shape'):
                        matrices.append(f"{m}: {mat.shape[0]}x{mat.shape[1]}")
                except:
                    continue
            
            if matrices:
                output.append("\n[MATRICES]")
                output.extend(matrices)
        except:
            pass

        # Add r() results
        r_results = EstimationResults.get_r_results()
        if r_results:
            output.append("\n[R RESULTS]")
            for name, value in r_results.items():
                if isinstance(value, (int, float)):
                    output.append(f"{name}: {value:.6g}")
                else:
                    output.append(f"{name}: {value}")
        
        # Add survey results
        svy_results = EstimationResults.get_svy_results()
        if svy_results:
            output.append("\n[SURVEY RESULTS]")
            for name, value in svy_results.items():
                output.append(f"{name}: {value:.6g}")
        
        # Add test results
        test_results = EstimationResults.get_test_results()
        if test_results:
            output.append("\n[TEST RESULTS]")
            for name, value in test_results.items():
                output.append(f"{name}: {value:.6g}")
        
        return "\n".join(output)

    @staticmethod
    def get_r_results() -> Dict[str, Any]:
        """Get r() results after non-estimation commands"""
        # Common r() scalars
        r_scalars = ['r(N)', 'r(sum)', 'r(mean)', 'r(sd)', 'r(Var)', 'r(min)', 'r(max)', 'r(sum_w)']
        # Common r() matrices
        r_matrices = ['r(table)', 'r(stats)', 'r(corr)', 'r(C)']
        
        results = {}
        for s in r_scalars:
            try:
                val = Scalar.getValue(s)
                if val is not None:
                    results[s] = val
            except:
                continue
                
        for m in r_matrices:
            try:
                mat = Matrix.get(m)
                if mat is not None:
                    results[m] = mat
            except:
                continue
                
        return results

    @staticmethod
    def get_svy_results() -> Dict[str, Any]:
        """Get survey estimation results"""
        svy_scalars = [
            'e(N_strata)', 'e(N_psu)', 'e(N_pop)',
            'e(df_r)', 'e(F_r)', 'e(F_p)',
            'e(N_strata_omit)', 'e(N_psu_omit)',
            'e(singleunit)', 'e(stage)'
        ]
        results = {}
        for s in svy_scalars:
            try:
                val = Scalar.getValue(s)
                if val is not None:
                    results[s] = val
            except:
                continue
        return results

    @staticmethod
    def get_test_results() -> Dict[str, Any]:
        """Get statistical test results more comprehensively"""
        # Expand test result scalars
        test_scalars = [
            # Basic test statistics
            'r(chi2)', 'r(p)', 'r(F)', 'r(df)', 'r(df_r)',
            # Additional test metrics
            'r(p_l)', 'r(p_u)', 'r(p_exact)',  # For exact tests
            'r(t)', 'r(se)', 'r(sd)',  # For t-tests
            'r(z)', 'r(level)',  # For z-tests
            'r(N_1)', 'r(N_2)', 'r(N_pair)',  # Sample sizes
            'r(mu_1)', 'r(mu_2)',  # Group means
            'r(rho)', 'r(corr)',  # Correlation tests
            # ANOVA specific
            'r(mss)', 'r(rss)', 'r(tss)',
            'r(r2)', 'r(r2_a)',
            # Additional test information
            'r(k)', 'r(n)', 'r(sum_w)'
        ]

        results = {}
        for s in test_scalars:
            try:
                val = Scalar.getValue(s)
                if val is not None:
                    results[s] = val
            except:
                continue

        # Also capture any test-specific matrices
        test_matrices = ['r(table)', 'r(chi2)', 'r(stats)']
        for m in test_matrices:
            try:
                mat = Matrix.get(m)
                if mat is not None:
                    results[f"{m}_matrix"] = mat
            except:
                continue

        return results

    @staticmethod
    def get_command_history() -> List[str]:
        """Get Stata command history using SFI"""
        try:
            history = []
            # First try to get the command line buffer size
            try:
                buffer_size = int(Macro.getGlobal("c(linebuffer)"))
            except:
                buffer_size = 20  # Default to 20 if we can't get the actual size
                
            # Get commands from line buffer (most recent first)
            for i in range(buffer_size):
                try:
                    # Use the same numbering as Stata's #review command
                    cmd = Macro.getGlobal(f"c(cmdline{i})")
                    if cmd and cmd.strip():
                        history.append(cmd.strip())
                except:
                    continue
                    
            return [cmd for cmd in history if cmd]  # Filter out empty commands
            
        except Exception as e:
            logger.warning(f"Error getting command history: {e}")
            return []

class ContextManager:
    def __init__(self):
        self.dataset_context: Optional[DatasetContext] = None
        self.command_history: List[str] = []
        self.session_state: Dict[str, Any] = {}
        self.estimation_results: Optional[Dict[str, Any]] = None
        self.chat_history: List[Dict[str, str]] = []  # Add chat history storage
        
    async def update_context(self, stata_output: str):
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
        """Update command history from Stata's actual command history"""
        self.command_history = EstimationResults.get_command_history()

# Core AI.do Assistant Class
class AiDoAssistant:
    def __init__(self, provider: ModelProvider):
        self.provider = provider
        self.context_manager = ContextManager()
        
    async def process_query(self, query: str) -> str:
        """Process query with automatic results inclusion and chat history"""
        # Add user query to chat history
        self.context_manager.add_chat_message("user", query)
        
        # Update command history from Stata
        self.context_manager.update_command_history()
        
        # Get base context with chat history
        context = self.context_manager.get_context_dict()
        formatted_variables = self._format_variables(context)
        
        # Get current estimation results
        results_text = EstimationResults.get_current_results()
        
        if results_text:
            self.context_manager.update_estimation_results({"text": results_text})
        
        context["recent_results"] = results_text
        
        formatted_prompt = self._get_prompt_template().format(
            query=query,
            context=context,
            formatted_variables=formatted_variables
        )
        
        # Get AI response
        response = await self.provider.generate_response(formatted_prompt, context)
        
        # Add AI response to chat history
        self.context_manager.add_chat_message("assistant", response)
        
        return response

    def _format_variables(self, context: Dict[str, Any]) -> str:
        """Optimized variable formatting with list comprehension"""
        metadata = context.get("dataset", {}).get("metadata", {}).get("columns", {})
        
        variables = [
            f"- {name} ({info['type']})"
            + (f": {info['label']}" if info['label'] else "")
            + (f" [Statistics: {', '.join(f'{k}={v:.2f}' for k, v in info['statistics'].items() if v is not None)}]" if info['statistics'] else "")
            + (f" [Labels: {', '.join(f'{k}={v}' for k, v in info['value_labels'].items())}]" if info['value_labels'] else "")
            for name, info in metadata.items()
        ]

        summary = f"""
        Dataset Summary:
        - Total Variables: {len(metadata)}
        - Numeric Variables: {sum(1 for info in metadata.values() if info['type'] not in ['str', 'strL'])}
        - String Variables: {sum(1 for info in metadata.values() if info['type'].startswith('str'))}
        - Variables with Labels: {sum(1 for info in metadata.values() if info['label'])}
        - Variables with Value Labels: {sum(1 for info in metadata.values() if info['value_labels'])}
        """
        
        return "\n".join(variables) + "\n" + summary

    @staticmethod
    def _get_prompt_template() -> str:
        """Updated template to include chat history"""
        return """
        Act as a Stata expert.
        Answer the following query using the provided dataset, estimation context, and chat history:

        Previous Messages:
        {context[chat_history]}

        Query: {query}

        Dataset Information:
        - Total Observations: {context[dataset][observations]}
        - Variables:
        {formatted_variables}

        Recent Results:
        {context[recent_results]}

        Command History:
        {context[history]}
        """
    
    async def update_dataset_context(self, stata_output: str):
        await self.context_manager.update_context(stata_output)

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