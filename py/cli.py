from aido import (
    AiDoAssistant,
    Config,
    ProviderFactory,
    DatasetContext,
    EstimationResults,
    ContextManager
)
import asyncio
from typing import Optional
import os  # Make sure os is imported

def parse_variable_info(varinfo: str) -> dict:
    """Parse the enhanced variable information string from Stata."""
    variables = {}
    for var_entry in varinfo.split('|'):
        if not var_entry.strip():
            continue
        try:
            parts = var_entry.split(':')
            if len(parts) != 8:  # Expect exactly 8 parts
                continue
                
            name, type_, label, mean, min_, max_, missing, value_labels = parts
            
            # Parse value labels more safely
            value_label_dict = {}
            if value_labels and value_labels != 'none':
                for vl in value_labels.split(';'):
                    if '=' in vl:
                        try:
                            key, value = vl.split('=', 1)
                            if key.strip() and value.strip():
                                value_label_dict[key.strip()] = value.strip()
                        except ValueError:
                            continue
                            
            variables[name.strip()] = {
                'type': type_.strip(),
                'label': label.strip(),
                'statistics': {
                    'mean': float(mean) if mean.strip() != '.' else None,
                    'min': float(min_) if min_.strip() != '.' else None,
                    'max': float(max_) if max_.strip() != '.' else None,
                    'missing': int(missing) if missing.strip() != '.' else 0
                },
                'value_labels': value_label_dict
            }
        except (ValueError, IndexError) as e:
            continue  # Skip malformed entries
            
    return variables

def parse_stata_results(results_info: str) -> dict:
    """Parse the Stata results string into a structured format."""
    results = {}
    if not results_info:
        return results
        
    for item in results_info.split('|'):
        if not item.strip():
            continue
        try:
            parts = item.split('=', 1)
            if len(parts) == 2:
                key, value = parts
                results[key.strip()] = value.strip()
        except Exception:
            continue
            
    return results

# Add global context manager
_context_manager: Optional['ContextManager'] = None

def process_stata_query(config_path: str, query: str, varinfo: str, 
                      typelist: str, nobs: int, results_file: str = ""):
    """Process a query from Stata with enhanced variable information and Stata results file"""
    async def main():
        try:
            global _context_manager
            
            # Load config and create provider
            config = Config(config_path)
            with open(config_path, 'r') as f:
                provider_name = f.readline().strip()
            provider = ProviderFactory.create_provider(provider_name, config)
            
            # Create or reuse context manager
            if _context_manager is None:
                _context_manager = ContextManager()
            
            # Create an AI.do Assistant instance with existing context
            assistant = AiDoAssistant(provider)
            assistant.context_manager = _context_manager
            
            # Parse the variable information
            variables = parse_variable_info(varinfo)
            
            # Update context with dataset information
            assistant.context_manager.dataset_context = DatasetContext(
                variables=list(variables.keys()),
                types=[v['type'] for v in variables.values()],
                observations=nobs,
                metadata={
                    "columns": variables,
                    "total_observations": nobs
                }
            )
            
            # Read results from file with better error handling
            results_text = ""
            print(f"Reading results from: {results_file}")  # Debug line
            
            if results_file:
                try:
                    if os.path.exists(results_file):
                        with open(results_file, 'r', encoding='utf-8') as f:
                            results_text = f.read()
                            print(f"Successfully read {len(results_text)} characters of results")  # Debug line
                    else:
                        print(f"Results file not found: {results_file}")
                except Exception as e:
                    print(f"Error reading results file: {e}")
                    # Continue with empty results rather than crashing
            
            # Store raw results text in session state
            assistant.context_manager.session_state["stata_results_raw"] = results_text
            
            # Process query and get response
            response = await assistant.process_query(query)
            print(response)
            
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(main())