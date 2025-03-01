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
        """Get ALL Stata results in a comprehensive, formatted way"""
        output = []
        
        # Get all e() and r() results using dynamic methods
        dynamic_results = EstimationResults.get_dynamic_results()
        if dynamic_results:
            output.append("\n[DYNAMIC RESULTS]")
            for name, value in sorted(dynamic_results.items()):
                if isinstance(value, (int, float)):
                    output.append(f"{name}: {value:.8g}")  # More decimal places
                else:
                    output.append(f"{name}: {value}")
        
        # Get all scalars
        scalars = EstimationResults.get_all_scalars()
        if scalars:
            output.append("\n[SCALARS]")
            for name, value in scalars.items():
                if value is not None:
                    output.append(f"{name}: {value:.8g}")  # More decimal places for precision
        
        # Get all macros
        macros = EstimationResults.get_all_macros()
        if macros:
            output.append("\n[MACROS]")
            for name, value in macros.items():
                if value:
                    output.append(f"{name}: {value}")
        
        # Get coefficient table with more detail
        coef_table = EstimationResults.get_coefficient_table()
        if coef_table and coef_table.get('coef'):  # Check if we have coefficients
            output.append("\n[COEFFICIENT TABLE]")
            output.append("Variable | Coef. | Std.Err. | t/z | P>|t/z| | [95% Conf. Interval]")
            output.append("-" * 80)
            
            for i, name in enumerate(coef_table['names']):
                try:
                    row = (f"{name:12} | {coef_table['coef'][i]:10.6f} | "  # More decimal places
                        f"{coef_table['se'][i]:10.6f} | {coef_table['t_stat'][i]:8.4f} | "  # More decimal places
                        f"{coef_table['p_value'][i]:8.6f} | [{coef_table['ci_lower'][i]:10.6f}, "  # More decimal places
                        f"{coef_table['ci_upper'][i]:10.6f}]")  # More decimal places
                    output.append(row)
                except:
                    continue
        
        # Get ALL matrices - don't limit to just names and dimensions
        try:
            # First try standard matrices
            matrix_names = [
                'e(b)', 'e(V)', 'e(ilog)', 'e(gradient)', 'e(S)', 'r(table)',
                # Add more potential matrices
                'e(Sigma)', 'r(R)', 'r(eigenvals)', 'r(eigenvectors)', 
                'r(loadings)', 'r(stats)'
            ]
            
            matrices = []
            matrix_contents = []
            
            for m in matrix_names:
                try:
                    mat = Matrix.get(m)
                    if mat is not None and hasattr(mat, 'shape'):
                        # Add matrix dimensions
                        matrices.append(f"{m}: {mat.shape[0]}x{mat.shape[1]}")
                        
                        # For smaller matrices, also include full contents
                        if mat.shape[0] * mat.shape[1] <= 100:  # Increased size threshold
                            content = [f"{m} contents:"]
                            # Try to get column names
                            try:
                                col_names = Matrix.getColNames(m)
                                if col_names:
                                    content.append("Column names: " + ", ".join(col_names))
                            except:
                                pass
                            
                            # Add matrix values
                            for i in range(min(mat.shape[0], 10)):  # Show up to 10 rows
                                row_vals = []
                                for j in range(min(mat.shape[1], 10)):  # Show up to 10 columns
                                    val = mat[i][j]
                                    if val is not None:
                                        row_vals.append(f"{val:.6g}")  # Format numbers
                                    else:
                                        row_vals.append(".")
                                content.append(f"Row {i+1}: [{', '.join(row_vals)}]")
                            
                            matrix_contents.extend(content)
                except:
                    continue
            
            if matrices:
                output.append("\n[MATRICES]")
                output.extend(matrices)
            
            if matrix_contents:
                output.append("\n[MATRIX CONTENTS]")
                output.extend(matrix_contents)
                
        except:
            pass

        # Add comprehensive r() results
        r_results = EstimationResults.get_r_results()
        if r_results:
            output.append("\n[R RESULTS]")
            output.append("Summary Statistics and Test Results:")
            
            # Sort r() results by name for better organization
            for name in sorted(r_results.keys()):
                value = r_results[name]
                if isinstance(value, (int, float)):
                    output.append(f"{name}: {value:.8g}")  # More decimal places
                else:
                    output.append(f"{name}: {value}")
        
        # Add survey results
        svy_results = EstimationResults.get_svy_results()
        if svy_results:
            output.append("\n[SURVEY RESULTS]")
            for name, value in svy_results.items():
                output.append(f"{name}: {value:.8g}")  # More decimal places
        
        # Add test results
        test_results = EstimationResults.get_test_results()
        if test_results:
            output.append("\n[TEST RESULTS]")
            for name, value in test_results.items():
                output.append(f"{name}: {value:.8g}")  # More decimal places

        # Add ALL available results using sfi
        all_avail_results = EstimationResults.get_all_available_results()
        if all_avail_results:
            output.append("\n[COMPLETE STATA RESULTS]")
            output.append("All Available Results from Stata:")
            for name in sorted(all_avail_results.keys()):
                if name not in dynamic_results and name not in scalars and name not in r_results:  # Use dynamic_results
                    value = all_avail_results[name]
                    if isinstance(value, (int, float)):
                        output.append(f"{name}: {value:.8g}")
                    else:
                        output.append(f"{name}: {value}")

        # Add this section after the test_results section
        # Add margins results
        margins_results = EstimationResults.get_margins_results()
        if margins_results:
            output.append("\n[MARGINS RESULTS]")
            output.append("Marginal Effects and Predictions:")
            for name, value in margins_results.items():
                if isinstance(value, (int, float)):
                    output.append(f"{name}: {value:.8g}")  # More decimal places
                else:
                    output.append(f"{name}: {value}")

        # Add Bayesian results
        bayesian_results = EstimationResults.get_bayesian_results()
        if bayesian_results:
            output.append("\n[BAYESIAN ESTIMATION RESULTS]")
            output.append("MCMC Diagnostics and Posterior Statistics:")
            for name, value in bayesian_results.items():
                if isinstance(value, (int, float)):
                    output.append(f"{name}: {value:.8g}")
                else:
                    output.append(f"{name}: {value}")

        # Add stored estimates info
        stored_estimates = EstimationResults.get_stored_estimates()
        if stored_estimates:
            output.append("\n[STORED ESTIMATES]")
            output.append(stored_estimates["stored_estimates"])

        # Add forecast results
        forecast_results = EstimationResults.get_forecast_results()
        if forecast_results:
            output.append("\n[FORECAST RESULTS]")
            for name, value in forecast_results.items():
                output.append(f"{name}: {value:.8g}")

        # Add SEM model results
        sem_results = EstimationResults.get_sem_results()
        if sem_results:
            output.append("\n[SEM RESULTS]")
            output.append("Structural Equation Model Fit Statistics:")
            for name, value in sem_results.items():
                if isinstance(value, (int, float)):
                    output.append(f"{name}: {value:.8g}")
                else:
                    output.append(f"{name}: {value}")

        # Add mixed model results
        mixed_results = EstimationResults.get_mixed_model_results()
        if mixed_results:
            output.append("\n[MIXED MODEL RESULTS]")
            output.append("Multilevel/Mixed Model Statistics:")
            for name, value in mixed_results.items():
                if isinstance(value, (int, float)):
                    output.append(f"{name}: {value:.8g}")
                else:
                    output.append(f"{name}: {value}")

        return "\n".join(output)

    @staticmethod
    def get_r_results() -> Dict[str, Any]:
        """Get COMPLETE r() results after non-estimation commands like summarize"""
        from sfi import Scalar, Matrix
        
        # Comprehensive list of r() scalars from ALL statistical commands
        r_scalars = [
            # Basic summary statistics (summarize)
            'r(N)', 'r(sum)', 'r(mean)', 'r(sd)', 'r(Var)', 'r(min)', 'r(max)', 'r(sum_w)',
            
            # Percentiles (summarize, detail)
            'r(p1)', 'r(p5)', 'r(p10)', 'r(p25)', 'r(p50)', 'r(p75)', 'r(p90)', 'r(p95)', 'r(p99)',
            
            # Additional detailed statistics
            'r(skewness)', 'r(kurtosis)', 'r(median)', 'r(iqr)', 'r(q)', 'r(range)', 
            
            # Test statistics (ttest)
            'r(N_1)', 'r(N_2)', 'r(p)', 'r(p_l)', 'r(p_u)', 'r(p_exact)',
            'r(t)', 'r(t_p)', 'r(df_t)', 'r(se)', 'r(sd_1)', 'r(sd_2)',
            'r(z)', 'r(z_p)', 'r(level)', 
            
            # Correlation and covariance (correlate)
            'r(rho)', 'r(cov_12)', 'r(corr)', 'r(cov)',
            
            # Tabulation results (tabulate)
            'r(r)', 'r(c)', 'r(chi2)', 'r(p_pearson)', 'r(gamma)', 'r(V)',
            'r(tau_a)', 'r(tau_b)', 'r(cramers_V)', 'r(expected)',
            
            # ANOVA and regression
            'r(mss)', 'r(rss)', 'r(tss)', 'r(F)', 'r(df_m)', 'r(df_r)',
            'r(r2)', 'r(r2_a)', 'r(rmse)',
            
            # Sample size information
            'r(k)', 'r(n)', 'r(sum_w)', 'r(N_strata)', 'r(N_psu)',
            
            # Return codes
            'r(rc)', 'r(level)',
            
            # Regression diagnostics
            'r(bgodfrey)', 'r(durbinalt)', 'r(dwstat)', 'r(rmse_rss)',
            
            # Factor analysis and PCA
            'r(f)', 'r(p_f)', 'r(rho_w)', 'r(df_m)', 'r(df_r)', 'r(chi2_p)',
            
            # Additional statistical tests
            'r(W)', 'r(V_sl)', 'r(df)', 'r(z_sl)', 'r(p_sl)', 'r(chi2_adj)',
            
            # Equality tests
            'r(drop)', 'r(df_r1)', 'r(df_r2)', 'r(f)', 'r(p_f)',
            
            # Various scalars with numerical suffixes (for pairs of statistics)
            'r(mu_1)', 'r(mu_2)', 'r(mu_diff)', 'r(se_diff)'
        ]
        
        # Dynamically try to get both normal and numbered variants (e.g., r(N), r(N1), etc.)
        results = {}
        
        # First get all explicit scalars in our comprehensive list
        for s in r_scalars:
            try:
                val = Scalar.getValue(s)
                if val is not None:
                    results[s] = val
            except:
                continue
        
        # Now add ALL numbered variants we might have missed
        for base in ['N_', 'mu_', 'sd_', 'se_', 'df_', 't_', 'z_', 'p_', 'chi2_', 'F_']:
            for i in range(1, 10):  # Try numbers 1-9
                try:
                    name = f"r({base}{i})"
                    val = Scalar.getValue(name)
                    if val is not None:
                        results[name] = val
                except:
                    continue
        
        # Get ALL matrices with detailed contents
        r_matrices = [
            'r(table)', 'r(stats)', 'r(corr)', 'r(C)', 'r(Cns)',
            'r(b)', 'r(se)', 'r(ci)', 'r(V)', 'r(R)',
            'r(means)', 'r(sums)', 'r(freqs)', 'r(p)', 'r(Sigma)',
            'r(S)', 'r(D)', 'r(Ev)', 'r(L)', 'r(lb)', 'r(ub)',
            'r(chol)', 'r(P)', 'r(T)', 'r(eigen)'
        ]
        
        for m in r_matrices:
            try:
                mat = Matrix.get(m)
                if mat is not None:
                    # Store complete matrix dimensions
                    if hasattr(mat, 'shape'):
                        results[f"{m}_dim"] = f"{mat.shape[0]}x{mat.shape[1]}"
                    
                    # For ALL matrices, store actual values regardless of size
                    # This ensures we capture everything
                    if hasattr(mat, 'shape'):
                        # Format the matrix in a structured way
                        matrix_str = "["
                        for i in range(min(mat.shape[0], 20)):  # Limit to first 20 rows if huge
                            row_vals = []
                            for j in range(min(mat.shape[1], 10)):  # Limit to first 10 cols if huge
                                val = mat[i][j]
                                if val is not None:
                                    row_vals.append(f"{val:.6g}")
                                else:
                                    row_vals.append(".")
                            matrix_str += "[" + ", ".join(row_vals) + "]"
                            if i < min(mat.shape[0], 20) - 1:
                                matrix_str += ", "
                        matrix_str += "]"
                        
                        # Store the formatted matrix
                        results[f"{m}_values"] = matrix_str
                        
                        # Also capture column names if available
                        try:
                            col_names = Matrix.getColNames(m)
                            if col_names:
                                results[f"{m}_colnames"] = col_names
                        except:
                            pass
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
        """Get ALL statistical test results without missing anything"""
        from sfi import Scalar, Matrix, Data
        
        # Expanded test result scalars - covers virtually everything
        test_scalars = [
            # Basic test statistics
            'r(chi2)', 'r(p)', 'r(F)', 'r(df)', 'r(df_r)',
            
            # Additional test metrics for ALL test types
            'r(p_l)', 'r(p_u)', 'r(p_exact)',  # For exact tests
            'r(t)', 'r(t_p)', 'r(se)', 'r(sd)',  # For t-tests
            'r(z)', 'r(z_p)', 'r(level)',  # For z-tests
            'r(N_1)', 'r(N_2)', 'r(N_pair)',  # Sample sizes
            'r(mu_1)', 'r(mu_2)', 'r(mu_diff)',  # Group means
            'r(rho)', 'r(corr)',  # Correlation tests
            
            # ANOVA specific
            'r(mss)', 'r(rss)', 'r(tss)',
            'r(r2)', 'r(r2_a)',
            
            # Additional test information
            'r(k)', 'r(n)', 'r(sum_w)',
            
            # Regression diagnostics - used by estat commands
            'r(bgodfrey)', 'r(durbinalt)', 'r(dwstat)',
            'r(estat)', 'r(p_estat)', 'r(scalar)',
            'r(w)', 'r(w_p)', 'r(h)', 'r(h_p)',
            
            # Prediction tests
            'r(pr2)', 'r(F_f)', 'r(p_f)',
            
            # Test statistics with numbered variants
            'r(df1)', 'r(df2)', 'r(stat1)', 'r(stat2)',
            'r(p1)', 'r(p2)', 'r(se1)', 'r(se2)'
        ]
        
        # ALSO check if we're after a test command by looking at command history
        try:
            # Try to get last command from Stata
            last_cmd = Data.execCommand("di \"`c(cmdline)'\"", False)
            if last_cmd and any(test_type in last_cmd.lower() for test_type in 
                               ['test', 'estat', 'hausman', 'lrtest', 'ttest', 
                                'pwcorr', 'correlate', 'tabulate']):
                # After a test command, try displaying and capturing ALL test results
                Data.execCommand("qui return list", False)
        except:
            pass

        results = {}
        
        # Get specific scalar values
        for s in test_scalars:
            try:
                val = Scalar.getValue(s)
                if val is not None:
                    results[s] = val
            except:
                continue
        
        # Try ALL possible numeric suffixes too (1-20)
        for base in ['chi2_', 'F_', 'p_', 'stat_', 'df_', 'w_', 't_', 'z_']:
            for i in range(1, 21):  # Try suffixes 1-20
                try:
                    name = f"r({base}{i})"
                    val = Scalar.getValue(name)
                    if val is not None:
                        results[name] = val
                except:
                    continue

        # Get test-specific matrices with FULL contents
        test_matrices = [
            'r(table)', 'r(chi2)', 'r(stats)', 'r(corr)',
            'r(Sigma)', 'r(V)', 'r(S)', 'r(b)', 'r(se)'
        ]
        
        for m in test_matrices:
            try:
                mat = Matrix.get(m)
                if mat is not None:
                    # Store dimensions
                    results[f"{m}_dim"] = f"{mat.shape[0]}x{mat.shape[1]}"
                    
                    # Store contents regardless of size
                    matrix_vals = []
                    for i in range(min(mat.shape[0], 10)):  # First 10 rows max
                        row = []
                        for j in range(min(mat.shape[1], 10)):  # First 10 cols max
                            try:
                                val = mat[i][j]
                                if val is not None:
                                    row.append(f"{val:.6g}")
                                else:
                                    row.append("null")
                            except:
                                row.append("error")
                        matrix_vals.append("[" + ",".join(row) + "]")
                    
                    results[f"{m}_values"] = "[" + ",".join(matrix_vals) + "]"
                    
                    # Try to get column names
                    try:
                        colnames = Matrix.getColNames(m)
                        if colnames:
                            results[f"{m}_colnames"] = colnames
                    except:
                        pass
            except:
                continue

        return results

    @staticmethod
    def get_command_history() -> List[str]:
        """Get Stata command history using SFI"""
        from sfi import Macro, Data
        
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
                    
            # If the above method fails, try a different approach - directly query Stata
            if not history:
                try:
                    cmd_output = Data.execCommand("qui log using _templist, replace text", False)
                    cmd_output = Data.execCommand("qui log off", False)
                    cmd_output = Data.execCommand("qui log using _templist, append text", False)
                    cmd_output = Data.execCommand("#review 15", True)
                    cmd_output = Data.execCommand("qui log close", False)
                    cmd_output = Data.execCommand("qui infix str cmd 1-244 using _templist if substr(ltrim(cmd),1,1)==\".", True)
                    cmd_output = Data.execCommand("qui cap erase _templist.log", False)
                    cmd_output = Data.execCommand("qui cap erase _templist.txt", False)
                    
                    # Parse command history if available
                    if cmd_output:
                        lines = cmd_output.strip().split("\n")
                        for line in lines:
                            if line.strip() and line.strip().startswith("."):
                                cmd = line.strip()[1:].strip()
                                if cmd:
                                    history.append(cmd)
                except:
                    pass
                    
            return [cmd for cmd in history if cmd]  # Filter out empty commands
            
        except Exception as e:
            import logging
            logging = logging.getLogger(__name__)
            logging.warning(f"Error getting command history: {e}")
            return []

    @staticmethod
    def get_dynamic_results() -> Dict[str, Any]:
        """Dynamically retrieve ALL available e() and r() results using ereturn list and return list"""
        from sfi import Data, Macro, Scalar
        results = {}
        
        # Get ALL e() results dynamically
        try:
            # Execute the ereturn list command and capture its output
            Data.execCommand("qui _return list, all", False)  # This shows ALL stored results
            
            # Try to get every possible e() scalar
            for prefix in ["e(", "r("]:
                # Start with common results we expect
                known_results = EstimationResults._get_common_results(prefix)
                if known_results:
                    for name, value in known_results.items():
                        results[name] = value
                
                # Then try to capture any additional results
                # We'll try a range of numbers that might be part of result names
                for i in range(1, 50):
                    for suffix in ["", "_aux", "_mi", "_eq"]:
                        for type_suffix in ["", "_p", "_se", "_ci", "_df"]:
                            try:
                                name = f"{prefix}{i}{suffix}{type_suffix})"
                                val = Scalar.getValue(name)
                                if val is not None:
                                    results[name] = val
                            except:
                                pass
        except Exception as e:
            pass  # Silently continue if this fails
        
        return results

    @staticmethod
    def _get_common_results(prefix: str) -> Dict[str, Any]:
        """Get common result patterns based on prefix"""
        from sfi import Scalar
        results = {}
        
        # Extended list of common results with all numeric suffixes
        suffixes = ["", "1", "2", "3", "4", "5"]
        base_names = [
            "N", "df", "df_m", "df_r", "F", "r2", "rmse", "mss", "rss", "ll", "ll_0", 
            "chi2", "p", "rank", "N_clust", "ic", "g", "g_min", "g_max", "g_avg", 
            "sigma", "sigma_e", "sigma_u", "r2_w", "r2_b", "r2_o", "rho", "N_g",
            "sum", "mean", "Var", "sd", "min", "max", "sum_w", "dev", 
            "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99",
            "skewness", "kurtosis", "median", "iqr", "q", "range",
            "t", "z", "rho", "cov", "corr", "N_strata", "N_psu", "N_pop", "rc", "level"
        ]
        
        for base in base_names:
            for suffix in suffixes:
                try:
                    name = f"{prefix}{base}{suffix})"
                    val = Scalar.getValue(name)
                    if val is not None:
                        results[name] = val
                except:
                    pass
                    
        return results

    @staticmethod
    def get_all_available_results() -> Dict[str, Any]:
        """Get ALL possible results using sfi's Data.execCommand to query Stata directly"""
        from sfi import Data, Scalar, Matrix
        results = {}
        
        try:
            # First execute a comprehensive return list command
            Data.execCommand("qui _return list, all", False)
            
            # Directly capture the output from return list
            r_output = Data.execCommand("return list", True)
            if r_output:
                # Parse the output to extract all scalars
                lines = r_output.strip().split('\n')
                for line in lines:
                    if ':' in line:
                        parts = line.strip().split(':', 1)
                        name = parts[0].strip()
                        try:
                            # Try to get this as a scalar
                            val = Scalar.getValue(name)
                            if val is not None:
                                results[name] = val
                        except:
                            # If not a scalar, try as a matrix
                            try:
                                mat_name = name.replace('r(', '').replace(')', '')
                                mat = Matrix.get(f"r({mat_name})")
                                if mat is not None and hasattr(mat, 'shape'):
                                    results[f"{name}_dim"] = f"{mat.shape[0]}x{mat.shape[1]}"
                                    # Add sample of values for smaller matrices
                                    if mat.shape[0] * mat.shape[1] <= 100:
                                        vals = []
                                        for i in range(min(mat.shape[0], 5)):
                                            for j in range(min(mat.shape[1], 5)):
                                                vals.append(f"{mat[i][j]:.6g}")
                                        results[f"{name}_sample"] = "[" + ", ".join(vals) + ", ...]"
                            except:
                                pass
            
            # Similarly for estimation results
            e_output = Data.execCommand("ereturn list", True)
            if e_output:
                # Parse the output to extract all scalars
                lines = e_output.strip().split('\n')
                for line in lines:
                    if ':' in line:
                        parts = line.strip().split(':', 1)
                        name = parts[0].strip()
                        try:
                            # Try to get this as a scalar
                            val = Scalar.getValue(name)
                            if val is not None:
                                results[name] = val
                        except:
                            # If not a scalar, try as a matrix
                            try:
                                mat_name = name.replace('e(', '').replace(')', '')
                                mat = Matrix.get(f"e({mat_name})")
                                if mat is not None and hasattr(mat, 'shape'):
                                    results[f"{name}_dim"] = f"{mat.shape[0]}x{mat.shape[1]}"
                                    # Add column names if available
                                    try:
                                        col_names = Matrix.getColNames(f"e({mat_name})")
                                        if col_names:
                                            results[f"{name}_colnames"] = col_names
                                    except:
                                        pass
                            except:
                                pass
                                
        except Exception as e:
            pass  # Silently continue if commands fail
        
        return results

    @staticmethod
    def get_margins_results() -> Dict[str, Any]:
        """Get comprehensive margins results after margins command"""
        from sfi import Scalar, Matrix, Data
        
        results = {}
        
        # Check if we've run a margins command
        try:
            # Try to detect if margins was the last command
            last_cmd = Data.execCommand("di \"`c(cmdline)'\"", False)
            is_margins = last_cmd and "margins" in last_cmd.lower()
            
            if is_margins:
                # Ensure results are accessible
                Data.execCommand("qui _return list, all", False)
            
            # Get margins-specific matrices
            margins_matrices = [
                'e(b_margins)', 'e(V_margins)', 'r(table)',
                'e(margins)', 'e(Jacobian)', 'e(error)'
            ]
            
            for m in margins_matrices:
                try:
                    mat = Matrix.get(m)
                    if mat is not None and hasattr(mat, 'shape'):
                        # Store dimensions
                        results[f"{m}_dim"] = f"{mat.shape[0]}x{mat.shape[1]}"
                        
                        # Get column names for interpretability
                        try:
                            col_names = Matrix.getColNames(m)
                            if col_names:
                                results[f"{m}_colnames"] = col_names
                        except:
                            pass
                        
                        # Store matrix values for smaller matrices
                        if mat.shape[0] * mat.shape[1] <= 100:
                            matrix_vals = []
                            for i in range(min(mat.shape[0], 10)):
                                row = []
                                for j in range(min(mat.shape[1], 10)):
                                    val = mat[i][j]
                                    if val is not None:
                                        row.append(f"{val:.6g}")
                                    else:
                                        row.append("null")
                                matrix_vals.append("[" + ",".join(row) + "]")
                            
                            results[f"{m}_values"] = "[" + ",".join(matrix_vals) + "]"
                except:
                    continue
            
            # Get margins-specific scalars
            margins_scalars = [
                'r(k_margins)', 'r(k_by)', 'r(k_at)', 
                'e(k_margins)', 'e(k_by)', 'e(k_at)',
                'r(level)', 'e(level)'
            ]
            
            for s in margins_scalars:
                try:
                    val = Scalar.getValue(s)
                    if val is not None:
                        results[s] = val
                except:
                    continue
                    
            # Capture at() and over() variables
            try:
                at_vars = Macro.getGlobal("e(atmeans)")
                if at_vars:
                    results["e(atmeans)"] = at_vars
            except:
                pass
                
            try:
                over_vars = Macro.getGlobal("e(over_vars)")
                if over_vars:
                    results["e(over_vars)"] = over_vars
            except:
                pass
                
        except Exception as e:
            pass  # Silently continue if commands fail
            
        return results

    @staticmethod
    def get_bayesian_results() -> Dict[str, Any]:
        """Get Bayesian estimation results after bayes commands"""
        # Fixed import statement to ensure Data is properly imported
        from sfi import Scalar, Matrix, Macro
        import sfi
        
        results = {}
        
        try:
            # First check if we have a Bayesian model using multiple methods
            cmd = Macro.getGlobal("e(cmd)")
            cmdline = Macro.getGlobal("e(cmdline)")
            prefix = Macro.getGlobal("e(prefix)")
            
            # Check for Bayesian commands in multiple ways
            is_bayes = (cmd and "bayes" in cmd.lower()) or \
                    (cmdline and "bayes" in cmdline.lower()) or \
                    (prefix and "bayes" in prefix.lower())
            
            # Try to directly check if specific Bayesian markers exist
            if not is_bayes:
                for marker in ["e(mcmcsize)", "e(dic)", "e(arate)"]:
                    try:
                        val = Scalar.getValue(marker)
                        if val is not None:
                            is_bayes = True
                            break
                    except:
                        continue
            
            if is_bayes:
                # Get all Bayesian scalars based on what we saw in the eret list output
                bayes_scalars = [
                    # MCMC parameters
                    "mcmcsize", "burnin", "mcmciter", "thinning", "nchains",
                    
                    # Convergence diagnostics
                    "eff_avg", "eff_min", "eff_max", "arate", 
                    
                    # Model criteria
                    "dic", "lml_lm", "clevel",
                    
                    # Prior specifications
                    "priorsigma", "priorshape", "priorscale",
                    
                    # Technical parameters
                    "blocksize", "block1_scale", "block2_scale",
                    "block1_tarate", "block2_tarate",
                    "adapt_tolerance", "adapt_alpha", "adapt_beta"
                ]
                
                for stat in bayes_scalars:
                    try:
                        val = Scalar.getValue(f"e({stat})")
                        if val is not None:
                            results[f"e({stat})"] = val
                    except:
                        pass
                
                # Get all critical Bayesian matrices - including key posterior statistics
                bayes_matrices = [
                    "mean",     # Posterior means
                    "sd",       # Posterior standard deviations
                    "mcse",     # Monte Carlo standard errors
                    "median",   # Posterior medians
                    "cri",      # Credible intervals
                    "ess",      # Effective sample sizes
                    "Cov",      # Posterior covariance
                    "init"      # Initial values
                ]
                
                for m in bayes_matrices:
                    try:
                        mat = Matrix.get(f"e({m})")
                        if mat is not None and hasattr(mat, 'shape'):
                            # Store dimensions
                            results[f"e({m})_dim"] = f"{mat.shape[0]}x{mat.shape[1]}"
                            
                            # Get parameter names if available
                            try:
                                col_names = Matrix.getColNames(f"e({m})")
                                if col_names:
                                    results[f"e({m})_colnames"] = col_names
                            except:
                                pass
                            
                            # Store matrix values
                            matrix_vals = []
                            for i in range(mat.shape[0]):
                                row = []
                                for j in range(min(mat.shape[1], 15)):  # Limit columns if very wide
                                    try:
                                        val = mat[i][j]
                                        if val is not None:
                                            row.append(f"{val:.6g}")
                                        else:
                                            row.append("null")
                                    except:
                                        row.append("error")
                                matrix_vals.append("[" + ", ".join(row) + "]")
                            
                            results[f"e({m})_values"] = "[" + ", ".join(matrix_vals) + "]"
                    except Exception as e:
                        continue
                
                # Capture Bayesian-specific macros without using Data.execCommand
                bayes_macros = [
                    "prefix", "method", "prior1", "prior2", "priorparams1", "priorparams2",
                    "likelihood", "block1_names", "block2_names", "parnames", "scparams",
                    "postvars", "pareqmap"
                ]
                
                for m in bayes_macros:
                    try:
                        val = Macro.getGlobal(f"e({m})")
                        if val:
                            results[f"e({m})"] = val
                    except:
                        continue
                    
                # If we have parameter names, format posterior summary as a table
                try:
                    param_names = Macro.getGlobal("e(parnames)")
                    mean_matrix = Matrix.get("e(mean)")
                    sd_matrix = Matrix.get("e(sd)")
                    
                    if param_names and mean_matrix is not None and sd_matrix is not None:
                        param_list = param_names.split()
                        results["posterior_table"] = "Parameter | Mean | Std. Dev.\n" + "-" * 40 + "\n"
                        
                        for i, param in enumerate(param_list):
                            if i < mean_matrix.shape[1] and i < sd_matrix.shape[1]:
                                mean_val = mean_matrix[0][i]
                                sd_val = sd_matrix[0][i]
                                results["posterior_table"] += f"{param} | {mean_val:.6g} | {sd_val:.6g}\n"
                except:
                    pass
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error getting Bayesian results: {str(e)}")
        
        return results

    @staticmethod
    def get_stored_estimates() -> Dict[str, Any]:
        """Get information about stored estimates"""
        from sfi import Data
        
        results = {}
        
        try:
            # Check if there are stored estimates
            estimates_list = Data.execCommand("estimates dir", True)
            if estimates_list and estimates_list.strip():
                results["stored_estimates"] = estimates_list
        except:
            pass
        
        return results

    @staticmethod
    def get_forecast_results() -> Dict[str, Any]:
        """Get time series forecasting results"""
        from sfi import Data, Scalar
        
        results = {}
        
        try:
            # Check for time series forecast command
            last_cmd = Data.execCommand("di \"`c(cmdline)'\"", False)
            if last_cmd and any(cmd in last_cmd.lower() for cmd in 
                            ["forecast", "arima", "var", "vecm"]):
                for metric in ["rmse", "mae", "mape", "ic"]:
                    try:
                        val = Scalar.getValue(f"e({metric})")
                        if val is not None:
                            results[f"e({metric})"] = val
                    except:
                        pass
        except:
            pass
        
        return results

    @staticmethod
    def get_sem_results() -> Dict[str, Any]:
        """Get Structural Equation Modeling results"""
        from sfi import Scalar, Matrix, Macro, Data
        
        results = {}
        try:
            # Multiple ways to detect SEM command
            cmd = Macro.getGlobal("e(cmd)")
            cmdline = Macro.getGlobal("e(cmdline)")
            is_sem = (cmd and cmd.lower() in ["sem", "gsem"]) or \
                    (cmdline and any(c in cmdline.lower() for c in ["sem", "gsem"]))
            
            if is_sem:
                # Ensure we get all results
                Data.execCommand("qui _return list, all", False)
                
                # Get SEM-specific fit statistics
                for stat in ["n_groups", "n_missing", "n_nobs", "n_parms",
                        "df_m", "chi2", "p", "cfi", "tli", "srmr", "cd", "rmsea",
                        "aic", "bic", "ic", "vce"]:
                    try:
                        val = Scalar.getValue(f"e({stat})")
                        if val is not None:
                            results[f"e({stat})"] = val
                    except:
                        pass
                        
                # Get path diagram information if available
                try:
                    # Try to get path coefficients
                    path_matrix = Matrix.get("e(b)")
                    if path_matrix is not None and hasattr(path_matrix, 'shape'):
                        results["sem_path_coefficients"] = "Available"
                        
                        # Get parameter names
                        try:
                            param_names = Matrix.getColNames("e(b)")
                            if param_names:
                                results["sem_parameters"] = ", ".join(param_names[:10]) + "..." if len(param_names) > 10 else ", ".join(param_names)
                        except:
                            pass
                except:
                    pass
        except Exception as e:
            import logging
            logging = logging.getLogger(__name__)
            logging.warning(f"Error getting SEM results: {e}")
            
        return results

    @staticmethod
    def get_mixed_model_results() -> Dict[str, Any]:
        """Get multilevel/mixed model results"""
        from sfi import Scalar, Matrix, Macro, Data
        
        results = {}
        try:
            cmd = Macro.getGlobal("e(cmd)")
            cmdline = Macro.getGlobal("e(cmdline)")
            
            is_mixed = (cmd and cmd.lower() in ["mixed", "xtmixed", "melogit", "meprobit", 
                                            "mepoisson", "menbreg", "xtmelogit", "xtmepoisson"]) or \
                    (cmdline and any(c in cmdline.lower() for c in 
                                    ["mixed", "me", "xtmixed", "xtme"]))
            
            if is_mixed:
                # Ensure we get all results
                Data.execCommand("qui _return list, all", False)
                
                # Get mixed model specific scalars
                for stat in ["N", "N_g", "g_min", "g_avg", "g_max", 
                        "ll", "ll_c", "chi2", "p", "rank", "k", "k_eq",
                        "ic", "converged", "rank0"]:
                    try:
                        val = Scalar.getValue(f"e({stat})")
                        if val is not None:
                            results[f"e({stat})"] = val
                    except:
                        pass
                        
                # Get group information
                try:
                    group_var = Macro.getGlobal("e(ivars)")
                    if group_var:
                        results["e(ivars)"] = group_var
                except:
                    pass
                
                # Get random effects matrices
                for m in ["b_s", "V_s", "sigma_u", "Sigma"]:
                    try:
                        mat = Matrix.get(f"e({m})")
                        if mat is not None and hasattr(mat, 'shape'):
                            results[f"e({m})_dim"] = f"{mat.shape[0]}x{mat.shape[1]}"
                    except:
                        continue
        except Exception as e:
            import logging
            logging = logging.getLogger(__name__)
            logging.warning(f"Error getting mixed model results: {e}")
            
        return results

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