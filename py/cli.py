from aido import (
    AiDoAssistant,
    Config,
    ProviderFactory,
    DatasetContext,
    EstimationResults,
    ContextManager
)
from typing import Optional
import os
import json
import tempfile
import math  # Add missing math import

def parse_variable_info(varinfo: str) -> dict:
    """Parse the enhanced variable information string from Stata."""
    variables = {}
    if not varinfo or varinfo.strip() == "":
        print("WARNING: Empty varinfo string received")
        return variables
        
    print(f"Parsing varinfo string with length: {len(varinfo)}")
    print(f"First 100 chars: {varinfo[:100]}")
    
    for var_entry in varinfo.split('|'):
        if not var_entry.strip():
            continue
        try:
            parts = var_entry.split(':')
            if len(parts) != 8:  # Expect exactly 8 parts
                print(f"Warning: Invalid variable entry format (expected 8 parts, got {len(parts)}): {var_entry[:50]}...")
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
            print(f"Error parsing variable entry: {e}")
            continue  # Skip malformed entries
            
    print(f"Successfully parsed {len(variables)} variables")
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

def get_dataset_sample(variable_names) -> dict:
    """Get a sample of the current Stata dataset using PyStata"""
    try:
        from sfi import Data, Characteristic
        
        # If no variables provided, get them directly from Stata
        if not variable_names:
            print("No variable names provided, trying to get them directly")
            try:
                var_count = Data.getVarCount()
                if var_count > 0:
                    variable_names = [Data.getVarName(i) for i in range(var_count)]
                    print(f"Retrieved {len(variable_names)} variables from Data.getVarCount()")
            except Exception as e:
                print(f"Error getting variable names: {e}")
        
        if not variable_names:
            print("ERROR: Could not retrieve any variable names")
            return {"rows": [], "error": "No variables found"}
            
        # Determine number of rows to sample (up to 20)
        total_obs = Data.getObsTotal()
        print(f"Dataset has {total_obs} total observations")
        sample_size = min(20, total_obs)
        
        # Create a sample data frame
        sample_data = {"rows": []}
        
        # Track missing value patterns
        missing_patterns = {}
        
        # Use Data.get() for more reliable bulk data retrieval when possible
        try:
            # Try to get data in bulk for efficiency
            data_matrix = Data.get(variable_names, range(sample_size))
            
            # Process the matrix into rows
            for row_idx, row_data in enumerate(data_matrix):
                row = {}
                for col_idx, var_name in enumerate(variable_names):
                    value = row_data[col_idx]
                    
                    # Handle Stata missing values
                    if isinstance(value, float):
                        # Check for Stata's special missing values
                        # In Stata, missing values are represented as very large numbers
                        # Standard missing is 8.988e+307, .a is 8.989e+307, .b is 8.990e+307, etc.
                        if value > 8.988e+307:
                            # Convert to Stata missing value representation
                            if value >= 8.989e+307 and value < 9.045e+307:
                                # Calculate which extended missing value (.a through .z)
                                mv_index = int(round((value - 8.989e+307) / 1.0e+305))
                                if 0 <= mv_index <= 25:  # .a through .z
                                    missing_char = chr(97 + mv_index)  # 'a' is 97 in ASCII
                                    value = f".{missing_char}"
                                    
                                    # Track missing pattern
                                    if var_name not in missing_patterns:
                                        missing_patterns[var_name] = {}
                                    if value not in missing_patterns[var_name]:
                                        missing_patterns[var_name][value] = 0
                                    missing_patterns[var_name][value] += 1
                            else:
                                # Standard missing
                                value = "."
                                
                                # Track missing pattern
                                if var_name not in missing_patterns:
                                    missing_patterns[var_name] = {}
                                if value not in missing_patterns[var_name]:
                                    missing_patterns[var_name][value] = 0
                                missing_patterns[var_name][value] += 1
                    
                    row[var_name] = value
                sample_data["rows"].append(row)
            
        except Exception as e:
            print(f"Bulk get failed: {e}, trying row-by-row approach")
            
            # Fallback to row-by-row approach
            for obs_idx in range(sample_size):
                row = {}
                for var in variable_names:
                    try:
                        # Try numeric first
                        value = Data.getNum(var, obs_idx)
                        
                        # Handle Stata missing values
                        if value is not None and isinstance(value, float):
                            if value > 8.988e+307:  # Stata missing value threshold
                                if value >= 8.989e+307 and value < 9.045e+307:
                                    # Extended missing (.a through .z)
                                    mv_index = int(round((value - 8.989e+307) / 1.0e+305))
                                    if 0 <= mv_index <= 25:
                                        value = f".{chr(97 + mv_index)}"
                                else:
                                    value = "."
                                    
                                # Track missing pattern
                                if var not in missing_patterns:
                                    missing_patterns[var] = {}
                                if value not in missing_patterns[var]:
                                    missing_patterns[var][value] = 0
                                missing_patterns[var][value] += 1
                                
                        # If value is None or NaN, it might be a string
                        if value is None or (isinstance(value, float) and math.isnan(value)):
                            value = Data.getStr(var, obs_idx)
                        
                        row[var] = value
                    except Exception as e:
                        print(f"Error getting value for {var}, obs {obs_idx}: {e}")
                        row[var] = None
                sample_data["rows"].append(row)
        
        # Add missing value patterns to sample data
        sample_data["missing_patterns"] = missing_patterns
        
        return sample_data
    except Exception as e:
        print(f"Error getting dataset sample: {e}")
        import traceback
        traceback.print_exc()
        return {"rows": [], "error": str(e)}

# Add global context manager
_context_manager: Optional['ContextManager'] = None

def process_stata_query(config_path: str, query: str, varinfo: str, 
                      typelist: str, nobs: int, results_file: str = ""):
    """Process a query from Stata with enhanced variable information and Stata results file"""
    try:
        global _context_manager
        
        print(f"\nProcessing query: '{query}'")
        print(f"Dataset has {nobs} observations")
        print(f"Variable info string length: {len(varinfo)}")
        
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
        
        # If no variables parsed but dataset has observations, 
        # try direct approach with Data API
        variable_names = list(variables.keys())
        if not variable_names and nobs > 0:
            try:
                from sfi import Data
                var_count = Data.getVarCount()
                if var_count > 0:
                    print(f"Direct access: found {var_count} variables")
                    # Get all variable names and types directly
                    direct_variables = {}
                    for i in range(var_count):
                        var_name = Data.getVarName(i)
                        var_type = Data.getVarType(i)
                        var_label = Data.getVarLabel(i)
                        
                        # Get missing count using misstable
                        missing_count = 0
                        try:
                            cmd_result = Data.execCommand(f"quietly misstable summarize {var_name}", False)
                            cmd_output = Data.execCommand(f"return list", True)
                            if "r(N_miss)" in cmd_output:
                                missing_str = cmd_output.split("r(N_miss)")[1].split("\n")[0].strip()
                                if "=" in missing_str:
                                    missing_count = int(missing_str.split("=")[1].strip())
                        except:
                            pass
                        
                        direct_variables[var_name] = {
                            'type': var_type,
                            'label': var_label,
                            'statistics': {'missing': missing_count},
                            'value_labels': {}
                        }
                    variables = direct_variables
                    variable_names = list(variables.keys())
                    print(f"Retrieved {len(variable_names)} variables directly from Stata")
            except Exception as e:
                print(f"Error retrieving variables directly: {e}")
        
        # Get dataset sample (first 20 rows)
        dataset_sample = get_dataset_sample(variable_names)
        
        # Update context with dataset information and sample
        assistant.context_manager.dataset_context = DatasetContext(
            variables=variable_names,
            types=[v['type'] for v in variables.values()],
            observations=nobs,
            metadata={
                "columns": variables,
                "total_observations": nobs,
                "sample_data": dataset_sample
            }
        )
        
        # Print dataset context with sample for debugging
        print("\n=== Dataset Context ===")
        print(f"Variables: {len(variable_names)}")
        print(f"Observations: {nobs}")
        
        # Print missing value information
        missing_vars = [name for name, info in variables.items() 
                       if info['statistics'].get('missing', 0) > 0]
        if missing_vars:
            print(f"Variables with missing values: {len(missing_vars)}")
            print(f"Examples: {', '.join(missing_vars[:5])}")
            
            # Show missing patterns if available
            if 'missing_patterns' in dataset_sample:
                print("Missing value patterns detected:")
                for var, patterns in list(dataset_sample['missing_patterns'].items())[:3]:
                    print(f"  {var}: {patterns}")
        
        # Print sample data
        print("Sample data (first few rows):")
        if dataset_sample["rows"]:
            for i, row in enumerate(dataset_sample["rows"][:3]):  # Show first 3 rows
                print(f"Row {i+1}: {json.dumps(row, default=str)}")
        else:
            print("No sample data available")
        print("======================\n")
        
        # Read results from file with better error handling
        results_text = ""
        if results_file:
            try:
                if os.path.exists(results_file):
                    with open(results_file, 'r', encoding='utf-8') as f:
                        results_text = f.read()
                else:
                    print(f"Results file not found: {results_file}")
            except Exception as e:
                print(f"Error reading results file: {e}")
        
        # Store raw results text in session state
        assistant.context_manager.session_state["stata_results_raw"] = results_text
        
        # Process query and get response
        response = assistant.process_query(query)
        print(response)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()