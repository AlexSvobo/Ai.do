program define aido
    version 16.0
    
    // Capture all user input into local
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

    // Initial progress display
    di as txt "["_continue
    forvalues i=1/10 {
        sleep 50
        di as res "▓" _continue
    }
    di as txt "]"

    // Define configuration and package paths
    local config_path "`c(sysdir_plus)'/a/aido.config"
    local package_path "`c(sysdir_plus)'/py"

    // Verify configuration exists
    capture confirm file "`config_path'"
    if _rc {
        di _newline as error "Configuration file not found at: `config_path'"
        exit 601
    }

    // Check that a dataset is loaded
    capture quietly describe
    if _rc != 0 {
        di _newline as error "╭{hline 48}╮"
        di as error "│" _col(3) "Error: No dataset currently in memory" _col(50) "│"
        di as error "│" _col(3) "Please load a dataset before running aido" _col(50) "│"
        di as error "╰{hline 48}╯"
        exit 4
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

    // Get variable names and types
    capture quietly ds
    local varnames "`r(varlist)'"
    
    // Initialize progress for variable processing
    local total_vars : word count `varnames'
    di as txt "Analyzing dataset variables:" as inp " `total_vars' variables found" _newline

    local vartypes
    local varinfo ""
    local i = 0
    local progress_char = "▋"
    local spinner "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    
    foreach v of local varnames {
        local ++i
        
        // Display progress bar
        if mod(`i', 5) == 0 {
            local pct = round(100 * `i'/`total_vars')
            di as txt "`progress_char'" as txt " `pct'%" _continue
            di char(13) _continue
        }

        local spinchar = substr("`spinner'", mod(`i',10)+1, 1)
        di as inp "`spinchar'" as txt " Processing " as res "`v'" as txt "..." _col(50) as inp "[`pct'%]" _continue
        di char(13) _continue

        // Get variable metadata
        local vartype : type `v'
        local vartypes "`vartypes' `vartype'"
        local varlabel : variable label `v'
        
        // Get basic statistics
        capture {
            quietly summarize `v', detail
            local mean = string(r(mean))
            local min = string(r(min))
            local max = string(r(max))
            local missing = string(r(N) - r(N_unique))
        }
        if _rc {
            local mean = "."
            local min = "."
            local max = "."
            local missing = "."
        }
        
        // Get value labels
        capture local vallab : value label `v'
        if !_rc & "`vallab'" != "" {
            local labelinfo ""
            capture {
                quietly levelsof `v', local(levels)
                foreach l of local levels {
                    local labval : label `vallab' `l'
                    local labelinfo "`labelinfo'`l'=`labval';"
                }
            }
        }
        else {
            local labelinfo "none"
        }
        
        // Build variable info string
        local varinfo "`varinfo'|`v':`vartype':`varlabel':`mean':`min':`max':`missing':`labelinfo'"
    }

    // Display preprocessing completion
    di as res "✓ " as txt "Preprocessing complete" _newline
    di as inp "→ " as txt "Sending query to AI.do..." _continue
    sleep 500
    di _newline

    // Call Python
    capture noisily {
        python: import sys; sys.path.insert(0, r"`package_path'"); from cli import process_stata_query; process_stata_query(r"`config_path'", "`finalquery'", "`varinfo'", r"`vartypes'", `=_N')
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