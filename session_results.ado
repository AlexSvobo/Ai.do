*! session_results.ado v1.4
*! Captures and saves r-class and e-class results to a text file with LLM-friendly formatting.

program define session_results
    version 16.0
    syntax, filename(string)

    // Drop any existing file handle with this name to avoid conflicts
    capture file close results_file
    
    // Create the file
    file open results_file using "`filename'", write replace text
    
    file write results_file "====== STATA SESSION RESULTS ======" _n _n
    
    // Capture and write r-class results
    file write results_file "====== R-CLASS RESULTS ======" _n _n
    
    // First check if _return has any content
    local has_returns = 0
    foreach rt in scalars macros matrices functions {
        capture local ret_items : r(`rt')
        if _rc==0 & "`ret_items'" != "" local has_returns = 1
    }
    
    if `has_returns' == 0 {
        file write results_file "No r-class results found." _n _n
    }
    else {
        // Process r-class scalars
        capture local r_scalars : r(scalars)
        if _rc==0 & "`r_scalars'" != "" {
            file write results_file "### R-CLASS SCALARS ###" _n
            foreach name of local r_scalars {
                capture noisily file write results_file "  r(`name') = " %20.6g (`=r(`name')') _n
            }
            file write results_file _n
        }
        
        // IMPROVED r-class macros handling
        capture local r_macros : r(macros)
        if _rc==0 & "`r_macros'" != "" {
            file write results_file "### R-CLASS MACROS ###" _n
            foreach name of local r_macros {
                // Try to get the value in a safe way
                capture local macval : di "`:di r(`name')'"
                if _rc == 0 & `"`macval'"' != "" {
                    // Successfully got a value
                    file write results_file "  r(`name') = `macval'" _n
                }
                else {
                    // Just show name without value
                    file write results_file "  r(`name') = " _n
                }
            }
            file write results_file _n
        }
        
        // Process r-class matrices with improved formatting
        capture local r_matrices : r(matrices)
        if _rc==0 & "`r_matrices'" != "" {
            file write results_file "### R-CLASS MATRICES ###" _n
            foreach name of local r_matrices {
                // Get dimensions directly
                capture {
                    local rows = rowsof(r(`name'))
                    local cols = colsof(r(`name'))
                    file write results_file "  r(`name') : `rows' x `cols' matrix" _n
                    
                    // Display actual matrix contents with labels
                    file write results_file "  Matrix contents:" _n
                    
                    // Get row and column names
                    local colnames 
                    local rownames
                    capture local colnames : colnames r(`name')
                    capture local rownames : rownames r(`name')
                    
                    // Check if symmetric
                    tempname __sym
                    capture mata: st_numscalar("`__sym'", issymmetric(st_matrix("r(`name')")))
                    if _rc==0 & scalar(`__sym')==1 {
                        file write results_file "  symmetric r(`name')[`rows',`cols']" _n
                    }
                    
                    // Write column headers with better spacing
                    file write results_file "  ROW_NAME  |"
                    if "`colnames'" != "" {
                        foreach colname of local colnames {
                            file write results_file %16s "`colname'"
                        }
                    }
                    else {
                        forvalues c = 1/`cols' {
                            file write results_file %16s "col_`c'"
                        }
                    }
                    file write results_file _n
                    
                    // Add separator line
                    file write results_file "  ----------|"
                    forvalues c = 1/`cols' {
                        file write results_file "----------------"
                    }
                    file write results_file _n
                    
                    // Write matrix values with row labels
                    forvalues r = 1/`rows' {
                        // Get row name
                        if "`rownames'" != "" {
                            local rowname : word `r' of `rownames'
                            file write results_file "  %-10s|" "`rowname'"
                        }
                        else {
                            file write results_file "  %-10s|" "row_`r'"
                        }
                        
                        // Write row values
                        forvalues c = 1/`cols' {
                            file write results_file %16.6f (`=r(`name')[`r',`c']')
                        }
                        file write results_file _n
                    }
                }
                file write results_file _n
            }
            file write results_file _n
        }
        
        // Process r-class functions
        capture local r_functions : r(functions)
        if _rc==0 & "`r_functions'" != "" {
            file write results_file "### R-CLASS FUNCTIONS ###" _n
            foreach name of local r_functions {
                file write results_file "  r(`name')" _n
            }
            file write results_file _n
        }
    }
    
    file write results_file "====== END R-CLASS RESULTS ======" _n _n
    
    // Capture e-class results
    file write results_file "====== E-CLASS RESULTS ======" _n _n
    
    // Check if e-class has any content
    local has_ereturns = 0
    foreach et in scalars macros matrices functions {
        capture local eret_items : e(`et')
        if _rc==0 & "`eret_items'" != "" local has_ereturns = 1
    }
    
    if `has_ereturns' == 0 {
        file write results_file "No e-class results found." _n _n
    }
    else {
        // Process e-class scalars
        capture local e_scalars : e(scalars)
        if _rc==0 & "`e_scalars'" != "" {
            file write results_file "### E-CLASS SCALARS ###" _n
            foreach name of local e_scalars {
                capture noisily file write results_file "  e(`name') = " %20.6g (`=e(`name')') _n
            }
            file write results_file _n
        }
        
        // IMPROVED e-class macros handling
        capture local e_macros : e(macros)
        if _rc==0 & "`e_macros'" != "" {
            file write results_file "### E-CLASS MACROS ###" _n
            foreach name of local e_macros {
                // Try to get the value in a safe way
                capture local macval : di "`:di e(`name')'"
                if _rc == 0 & `"`macval'"' != "" {
                    // Successfully got a value
                    file write results_file "  e(`name') = `macval'" _n
                }
                else {
                    // Just show name without value
                    file write results_file "  e(`name') = " _n
                }
            }
            file write results_file _n
        }
        
        // Process e-class matrices with improved formatting
        capture local e_matrices : e(matrices)
        if _rc==0 & "`e_matrices'" != "" {
            file write results_file "### E-CLASS MATRICES ###" _n
            foreach name of local e_matrices {
                // Get dimensions directly
                capture {
                    local rows = rowsof(e(`name'))
                    local cols = colsof(e(`name'))
                    file write results_file "  e(`name') : `rows' x `cols' matrix" _n
                    
                    // Display actual matrix contents with labels
                    file write results_file "  Matrix contents:" _n
                    
                    // Get row and column names
                    local colnames 
                    local rownames
                    capture local colnames : colnames e(`name')
                    capture local rownames : rownames e(`name')
                    
                    // Check if symmetric
                    tempname __sym
                    capture mata: st_numscalar("`__sym'", issymmetric(st_matrix("e(`name')")))
                    if _rc==0 & scalar(`__sym')==1 {
                        file write results_file "  symmetric e(`name')[`rows',`cols']" _n
                    }
                    
                    // Write column headers with better spacing
                    file write results_file "  ROW_NAME  |"
                    if "`colnames'" != "" {
                        foreach colname of local colnames {
                            file write results_file %16s "`colname'"
                        }
                    }
                    else {
                        forvalues c = 1/`cols' {
                            file write results_file %16s "col_`c'"
                        }
                    }
                    file write results_file _n
                    
                    // Add separator line
                    file write results_file "  ----------|"
                    forvalues c = 1/`cols' {
                        file write results_file "----------------"
                    }
                    file write results_file _n
                    
                    // Write matrix values with row labels
                    forvalues r = 1/`rows' {
                        // Get row name
                        if "`rownames'" != "" {
                            local rowname : word `r' of `rownames'
                            file write results_file "  %-10s|" "`rowname'"
                        }
                        else {
                            file write results_file "  %-10s|" "row_`r'"
                        }
                        
                        // Write row values
                        forvalues c = 1/`cols' {
                            file write results_file %16.6f (`=e(`name')[`r',`c']')
                        }
                        file write results_file _n
                    }
                }
                file write results_file _n
            }
            file write results_file _n
        }
        
        // Process e-class functions
        capture local e_functions : e(functions)
        if _rc==0 & "`e_functions'" != "" {
            file write results_file "### E-CLASS FUNCTIONS ###" _n
            foreach name of local e_functions {
                file write results_file "  e(`name')" _n
            }
            file write results_file _n
        }
    }
    
    file write results_file "====== END E-CLASS RESULTS ======" _n _n
    file write results_file "====== END STATA SESSION RESULTS ======" _n

    // Close the file
    file close results_file
    
    di as text "Session results saved to `filename'"
end