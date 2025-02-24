program define aido_config, rclass
    version 16.0
    syntax, provider(string) apikey(string) [modelid(string)]

    * Define config file path
    local config_path "`c(sysdir_plus)'/a/aido.config"

    * Save settings to a permanent Stata characteristics
    char _dta[AIDO_PROVIDER] "`provider'"
    char _dta[AIDO_APIKEY] "`apikey'"
    
    if "`modelid'" != "" {
        char _dta[AIDO_MODELID] "`modelid'"
    }

    * Save to disk for persistence between Stata sessions
    quietly {
        * Create or clear existing config file
        tempname fh
        file open `fh' using "`config_path'", write replace
        file write `fh' "`provider'" _n
        file write `fh' "`apikey'" _n
        if "`modelid'" != "" {
            file write `fh' "`modelid'" _n
        }
        file close `fh'
    }

    * Confirm settings were saved
    di "Ai.do Configuration Updated:"
    di "Provider: `provider'"
    di "API Key: [Hidden for security]"
    if "`modelid'" != "" {
        di "Model ID: `modelid'"
    }
end
