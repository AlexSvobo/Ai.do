program define aido
    version 16.0
    
    // Continue with rest of program
    args anything
    if "`anything'" == "" {
        di as error "❌ Error: Query cannot be empty"
        exit 198
    }
    local finalquery "`anything'"
    
    // Create a stylized header
    di as txt "{hline 50}"
    di as txt "╭{hline 48}╮"
    di as txt "│" _col(18) as res "AI.do Query" _col(50) as txt "│"
    di as txt "│" _col(20) as res "Processing" _col(50) as txt "│"
    di as txt "╰{hline 48}╯"
    di as txt "{hline 50}" _newline

    // Define configuration and package paths
    local config_path "`c(sysdir_plus)'/a/aido.config"
    local package_path "`c(sysdir_plus)'/py"

    // Verify configuration exists
    capture confirm file "`config_path'"
    if _rc {
        di _newline as error "Configuration file not found at: `config_path'"
        exit 601
    }
    
    
    // Create a temporary file for the query
    tempname queryfile
    tempfile queryfilename
    capture {
        quietly file open `queryfile' using `"`queryfilename'"', write replace
        file write `queryfile' "`finalquery'"
        file close `queryfile'
    }
    if _rc {
        di _newline as error "Error creating temporary query file"
        exit _rc
    }


    // Display preprocessing completion
    di as res "✓ " as txt "Preprocessing complete" _newline
    
    // Create a temporary file for storing the results
    tempfile results_file
    
    // Drop any existing file handle with this name to avoid conflicts
    capture file close aido_results
    
    // Create the file for storing results
    file open aido_results using "`results_file'", write replace text
    
    file write aido_results "====== STATA SESSION RESULTS ======" _n _n
    
    // Write dataset info directly (without calling describe/summarize again)
    file write aido_results "====== DATASET CONTEXT ======" _n _n
    file write aido_results "Total observations: `=_N'" _n
    file write aido_results "Variables: `total_vars'" _n
    if "`c(filename)'" != "" file write aido_results "Dataset: `c(filename)'" _n
    file write aido_results _n
    

    
    // Simplify r-class results capture - just get what's currently available
    file write aido_results "====== R-CLASS RESULTS ======" _n _n
    
    // Check if there are any r-class results
    capture local r_scalars : r(scalars)
    capture local r_macros : r(macros)
    capture local r_matrices : r(matrices)
    capture local r_functions : r(functions)
    
    if _rc==0 & ("`r_scalars'`r_macros'`r_matrices'`r_functions'" != "") {
        // Process r-class scalars
        if "`r_scalars'" != "" {
            file write aido_results "### R-CLASS SCALARS ###" _n
            foreach name of local r_scalars {
                capture noisily file write aido_results "  r(`name') = " %20.6g (`=r(`name')') _n
            }
            file write aido_results _n
        }
        
        // Process r-class macros
        if "`r_macros'" != "" {
            file write aido_results "### R-CLASS MACROS ###" _n
            foreach name of local r_macros {
                capture local macval : di "`:di r(`name')'"
                if _rc == 0 {
                    file write aido_results "  r(`name') = `macval'" _n
                }
            }
            file write aido_results _n
        }
        
        // Process r-class matrices (simplified)
        if "`r_matrices'" != "" {
            file write aido_results "### R-CLASS MATRICES ###" _n
            foreach name of local r_matrices {
                file write aido_results "  r(`name') matrix" _n
            }
            file write aido_results _n
        }
        
        // Process r-class functions
        if "`r_functions'" != "" {
            file write aido_results "### R-CLASS FUNCTIONS ###" _n
            foreach name of local r_functions {
                file write aido_results "  r(`name')" _n
            }
            file write aido_results _n
        }
    }
    else {
        file write aido_results "No r-class results found." _n _n
    }
    
    file write aido_results "====== END R-CLASS RESULTS ======" _n _n
    
    // Keep the e-class handling as is since it works well
    file write aido_results "====== E-CLASS RESULTS ======" _n _n
    
    // Check if e-class has any content
    local has_ereturns = 0
    foreach et in scalars macros matrices functions {
        capture local eret_items : e(`et')
        if _rc==0 & "`eret_items'" != "" local has_ereturns = 1
    }
    
    if `has_ereturns' == 0 {
        file write aido_results "No e-class results found." _n _n
    }
    else {
        // Process e-class scalars
        capture local e_scalars : e(scalars)
        if _rc==0 & "`e_scalars'" != "" {
            file write aido_results "### E-CLASS SCALARS ###" _n
            foreach name of local e_scalars {
                capture noisily file write aido_results "  e(`name') = " %20.6g (`=e(`name')') _n
            }
            file write aido_results _n
        }
        
        // e-class macros handling
        capture local e_macros : e(macros)
        if _rc==0 & "`e_macros'" != "" {
            file write aido_results "### E-CLASS MACROS ###" _n
            foreach name of local e_macros {
                // Try to get the value in a safe way
                capture local macval : di "`:di e(`name')'"
                if _rc == 0 & `"`macval'"' != "" {
                    // Successfully got a value
                    file write aido_results "  e(`name') = `macval'" _n
                }
                else {
                    // Just show name without value
                    file write aido_results "  e(`name') = " _n
                }
            }
            file write aido_results _n
        }
        
        // Process e-class matrices
        capture local e_matrices : e(matrices)
        if _rc==0 & "`e_matrices'" != "" {
            file write aido_results "### E-CLASS MATRICES ###" _n
            foreach name of local e_matrices {
                // Get dimensions directly
                capture {
                    local rows = rowsof(e(`name'))
                    local cols = colsof(e(`name'))
                    file write aido_results "  e(`name') : `rows' x `cols' matrix" _n
                    
                    // Display actual matrix contents with labels
                    file write aido_results "  Matrix contents:" _n
                    
                    // Get row and column names
                    local colnames 
                    local rownames
                    capture local colnames : colnames e(`name')
                    capture local rownames : rownames e(`name')
                    
                    // Check if symmetric
                    tempname __sym
                    capture mata: st_numscalar("`__sym'", issymmetric(st_matrix("e(`name')")))
                    if _rc==0 & scalar(`__sym')==1 {
                        file write aido_results "  symmetric e(`name')[`rows',`cols']" _n
                    }
                    
                    // Write column headers with better spacing
                    file write aido_results "  ROW_NAME  |"
                    if "`colnames'" != "" {
                        foreach colname of local colnames {
                            file write aido_results %16s "`colname'"
                        }
                    }
                    else {
                        forvalues c = 1/`cols' {
                            file write aido_results %16s "col_`c'"
                        }
                    }
                    file write aido_results _n
                    
                    // Add separator line
                    file write aido_results "  ----------|"
                    forvalues c = 1/`cols' {
                        file write aido_results "----------------"
                    }
                    file write aido_results _n
                    
                    // Write matrix values with row labels
                    forvalues r = 1/`rows' {
                        // Get row name
                        if "`rownames'" != "" {
                            local rowname : word `r' of `rownames'
                            file write aido_results "  %-10s|" "`rowname'"
                        }
                        else {
                            file write aido_results "  %-10s|" "row_`r'"
                        }
                        
                        // Write row values
                        forvalues c = 1/`cols' {
                            file write aido_results %16.6f (`=e(`name')[`r',`c']')
                        }
                        file write aido_results _n
                    }
                }
                file write aido_results _n
            }
            file write aido_results _n
        }
        
        // Process e-class functions
        capture local e_functions : e(functions)
        if _rc==0 & "`e_functions'" != "" {
            file write aido_results "### E-CLASS FUNCTIONS ###" _n
            foreach name of local e_functions {
                file write aido_results "  e(`name')" _n
            }
            file write aido_results _n
        }
    }
    
    file write aido_results "====== END E-CLASS RESULTS ======" _n _n
    file write aido_results "====== END STATA SESSION RESULTS ======" _n

    // Close the file
    file close aido_results
    
    di as res "✓" _newline
    di as inp "→ " as txt "Sending query to AI.do..." _continue
    sleep 500
    di _newline

    // Call Python with the filename instead of the content
    capture noisily {
        // Pass the file path rather than file contents
        python: import sys; sys.path.insert(0, r"`package_path'"); from cli import process_stata_query; process_stata_query(r"`config_path'", "`finalquery'", "`varinfo'", r"`vartypes'", `=_N', r"`results_file'")
    }
    if _rc {
        di _newline as error "╭{hline 48}╮"
        di as error "│" _col(3) "Error in Python processing" _col(50) "│"
        di as error "│" _col(3) "Please check Python installation and dependencies" _col(50) "│"
        di as error "╰{hline 48}╯"
        exit _rc
    }
    
    // Final success message
    local end_time = c(current_time)
    di _newline
    di as txt "╭{hline 48}╮"
    di as txt "│" _col(2) as res "✓ " as inp "Process completed successfully at: " as res "`end_time'" _col(50) as txt "│"
    di as txt "╰{hline 48}╯"
end