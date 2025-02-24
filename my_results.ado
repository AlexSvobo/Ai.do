program define my_results, rclass
    version 18.0
    // Display key results
    display "Dependent variable: " e(depvar)
    display "Number of observations: " e(N)
    display "R-squared: " e(r2)
    matrix list e(b)
    
    // Initialize result string
    local eresults ""
    
    // Manually define list of common scalars from e() results
    local scalars "N df_m df_r F r2 rmse mss rss r2_a"
    foreach s of local scalars {
         capture scalar tmp = e(`s')
         if !_rc {
             local val : display %9.3f e(`s')
             local eresults "`eresults'`s': `val' | "
         }
    }
    
    // Manually define list of common macros from e() results
    local macros "depvar cmd cmdline"
    foreach m of local macros {
         if ("`e(`m')'" != "") {
             local val "`e(`m')'"
             local eresults "`eresults'`m': `val' | "
         }
    }
    
    // Capture coefficients and standard errors for each predictor
    matrix b = e(b)
    local varlist : colnames b
    foreach var of local varlist {
         local coef = _b[`var']
         local se   = _se[`var']
         local coef_str : display %9.3f `coef'
         local se_str   : display %9.3f `se'
         local eresults "`eresults'`var': b=`coef_str'; se=`se_str' | "
    }
    

    
    // Return eresults as a returned local
    return local eresults `"`eresults'"'
    
    // Display the final concatenated result string
    display as text "`eresults'"
end