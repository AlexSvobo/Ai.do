# AI.do Assistant

An AI-powered assistant for Stata.

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Open Source](https://img.shields.io/badge/Open%20Source-Yes-green.svg)]()
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Stata 16+](https://img.shields.io/badge/Stata-16%2B-blue.svg)](https://www.stata.com)
[![Requires aiohttp](https://img.shields.io/badge/Requires-aiohttp-brightgreen.svg)](requirements.txt)
[![GitHub Issues](https://img.shields.io/github/issues/alexsvobo/Ai.do)](https://github.com/alexsvobo/Ai.do/issues)

## Requirements

- Stata 16.0 or later
- Python 3.8 or later (we'll help you install it)
- API key from one of these providers:
  - [Google Gemini (recommended)](https://makersuite.google.com/app/apikey)
  - [Hugging Face](https://huggingface.co/settings/tokens)

## Installation

1. **Install Python** (if not already installed):
   - Download Python from [python.org](https://www.python.org/downloads/)
   - During installation:
     - ✓ Check "Add Python to PATH"
     - ✓ Check "Install pip"
   - Verify installation by opening Stata and typing:
```stata
python query
```
   If you see a Python version number, you're good to go!

2. **Find your Stata PLUS directory**:
```stata
sysdir
```
Look for the path next to "PLUS:". It's typically:
- Windows: `c:/users/username/ado/plus`
- Mac: `/Users/username/Library/Application Support/Stata/ado/plus`
- Linux: `~/ado/plus`

3. **Install AI.do Assistant**:
```stata
net install aido, from("https://raw.githubusercontent.com/alexsvobo/Ai.do/main/") replace
```

4. **Install Python package**:
   - Windows: Press Windows+R, type "cmd", press Enter, then type:
```cmd
pip install aiohttp
```
   - Mac/Linux: Open Terminal and type:
```bash
pip3 install aiohttp
```

5. **Configure your AI provider** in Stata:
```stata
// For Google Gemini (recommended):
aido_config, provider(gemini) apikey("YOUR-API-KEY")

// For Hugging Face:
aido_config, provider(huggingface) apikey("YOUR-API-KEY") modelid("Qwen/Qwen2.5-Coder-32B")
```

## Quick Start

1. Load any dataset:
```stata
sysuse auto
```

2. Ask a question:
```stata
aido "write a report on these results"
```

3. Dialog Box may be used as UI:
```stata
db aido
```

## Features

- Database context
- Session results context (soon)
- Multiple AI provider support

## Support

Need help? Open an issue on our [GitHub page](https://github.com/alexsvobo/Ai.do/issues)

## License

MIT License - See LICENSE file for details

## Notes

This code initializes PyStata, a Python-Stata bridge included with Stata. A full Python installation is required to run it. When making your first AI request, the Stata startup message will appear in the results window—this automatic behavior confirms PyStata has been successfully initialized.

## Example Session
```markdown

. aido "prove to me that you know our conversational context.
--------------------------------------------------
╭------------------------------------------------╮
│                AI.do Query                     │
│                  Processing                    │
╰------------------------------------------------╯
--------------------------------------------------

[▓▓▓▓▓▓▓▓▓▓]
Analyzing dataset variables: 12 variables found

� Processing make...                             [%]
� Processing price...                            [%]
� Processing mpg...                              [%]
� Processing rep78...                            [%]
▋ 42%
� Processing headroom...                         [42%]
� Processing trunk...                            [42%]
� Processing weight...                           [42%]
� Processing length...                           [42%]
� Processing turn...                             [42%]
▋ 83%
� Processing displacement...                     [83%]
� Processing gear_ratio...                       [83%]
� Processing foreign...                          [83%]
✓ Preprocessing complete

→ Sending query to AI.do...

Okay, I can demonstrate my awareness of the conversational context. Here's how I'll prove it, drawing from our previous interactions and the information you'
> ve now provided:

1.  **Awareness of Past Instructions:** I remember that you initially prompted me with the command `more` when there was no data loaded. I correctly identifi
> ed that the "more" command would be ineffective with an empty dataset and advised you to load data first. This shows I understand Stata's basic commands an
> d their limitations.

2.  **The "Prove It" Thread:** I recall you asking me to prove I could be a useful Stata assistant in a professional sense. In response, I provided a detaile
> d breakdown of my capabilities in data management, statistical analysis, programming, data visualization, and ensuring reproducibility. I emphasized the im
> portance of do-files, which is crucial for professional Stata use.

3.  **Bayesian Challenge:** I remember that you specifically requested that I showcase my expert-level Stata coding skills, including Bayesian analysis. I pr
> ovided a detailed Stata do-file demonstrating a Bayesian linear regression with informative priors, MCMC diagnostics, and posterior predictive checking. I 
> highlighted the importance of informative priors and model diagnostics, reflecting a deeper understanding of Bayesian principles.

4.  **Current Dataset:** I understand that you have now loaded the `auto.dta` dataset.  I can see from your description that it contains information about di
> fferent car models, including variables like `price`, `mpg`, `weight`, and `foreign`.

5.  **Recent Results:** I know the last result returned `r(N): 74`, which means that the most recent command was likely one that returned the number of obser
> vations.

6.  **Building on Previous Bayesian Example, Addressing the Specific Dataset:**  Given the context, I'll now show how the *previous* Bayesian code could be a
> dapted to the loaded dataset, *and* I'll incorporate what I know about the data:

```stata
/*
  Expert-Level Stata Code: Bayesian Linear Regression with Informative Priors
  (Adapted for auto.dta)
*/

clear all //remove all variables and free all space
set more off // turn off the --more--

sysuse auto, clear // Load the auto dataset (redundant, but good practice)

// 1. Data Preparation (Addressing Potential Issues and Scaling)

// Check for missing values in the key variables.  Act on them *before* bayes.
misstable summarize price weight foreign mpg rep78 headroom trunk length turn displacement gear_ratio

// Since 'rep78' has missing values, we'll drop observations with missing values
// A more sophisticated approach might involve imputation, but that is beyond the scope of this example.
drop if missing(rep78)

// Scale weight to improve MCMC convergence.
generate weight_scaled = weight / 1000  // Weight in thousands of lbs.
label variable weight_scaled "Weight (thousands of lbs)"

// Consider standardizing mpg as well:
generate mpg_scaled = (mpg - rmean(mpg)) / rsd(mpg)
label variable mpg_scaled "MPG (Standardized)"


// 2. Define Informative Priors (Crucial for Expert-Level Bayesian Analysis)

// priors
// The example below is only illustrative.
// The user should choose parameters based on their prior experience.
// and/or use the reference data.
summarize price weight foreign mpg
display r(mean)
display r(sd)

// Example Priors Based on the 'auto.dta' Data Itself:
//   - We'll use information *from* the dataset to inform our priors, but this is purely illustrative.
//   - In a real analysis, you'd use *external* prior knowledge.
//   - This is just to show how to adapt the priors.
local price_mean = r(mean)
local price_sd   = r(sd)
local weight_mean = r(mean)
local weight_sd   = r(sd)

// 3. Bayesian Linear Regression Model

bayes,  rseed(12345) mcmcsize(10000) burnin(2500)  prior({price: weight_scaled}, normal(`weight_mean'/1000, `weight_sd'/1000)) ///
       prior({price: foreign}, normal(0, `price_sd'/2)) ///
       prior({price:mpg_scaled}, normal(0, `price_sd'/2)) ///
       prior({price:_cons}, normal(`price_mean`, `price_sd`)) ///
       prior(sigma2, invgamma(2, 100000)): regress price weight_scaled foreign mpg_scaled

// 4. Model Diagnostics (Essential for Ensuring MCMC Convergence)

// Trace plots:  Visually inspect the MCMC chains for stationarity.
bayesgraph diagnostics weight_scaled foreign mpg_scaled _cons sigma2, title("Trace Plots")

// Autocorrelation plots:  Check for autocorrelation in the MCMC chains.
bayesgraph ac weight_scaled foreign mpg_scaled _cons sigma2, title("Autocorrelation Plots")

// Histograms and Kernel Density Plots of the Posterior Distributions
bayesgraph histogram weight_scaled foreign mpg_scaled _cons sigma2, title("Posterior Histograms")
bayesgraph kdensity weight_scaled foreign mpg_scaled _cons sigma2, title("Posterior Density Plots")

// 5. Summarize Results (Posterior Means, Credible Intervals)
bayesstats summary  // Display posterior means, standard deviations, and credible intervals
bayesstats ess       // Effective Sample Size:  Should be reasonably large (rule of thumb: > 200)
bayesstats pvalues    // Posterior probabilities of being greater or less than zero


// Posterior Predictive Checking
//Simulate draws from the posterior predictive distribution:
simulate price_pred = rnormal(_b[_cons] + _b[weight_scaled]*weight_scaled + _b[foreign]*foreign + _b[mpg_scaled]*mpg_scaled, sqrt(_b[sigma2])), reps(500)

// Compare the distribution of the predicted prices to the actual prices.
histogram price,  title("Actual Price")
histogram price_pred, title("Posterior Predictive Price")


// Test the hypothesis that the coefficient on \'weight_scaled\' is greater than 0.\
bayesstats hypothesis {weight_scaled > 0}


**Key Demonstrations of Context Awareness:**

*   **`sysuse auto, clear`**:  While technically redundant (because the environment indicates the data is loaded), this shows I know *the name of the dataset
> * you're using.
*   **Addressing Missing Data:**  The code now *explicitly* checks for missing values in multiple relevant variables using `misstable summarize`.  Crucially,
>  it then *removes* observations with missing `rep78`, showing I understand data cleaning is essential, even in demonstration code.  I acknowledge that a mo
> re advanced approach *could* impute missing data.
*   **Adaptation of Priors:** The code now uses the `summarize` command to *extract* the mean and standard deviation of `price` and `weight` from the loaded 
> dataset. It then uses these values to *inform* the priors.  This demonstrates that I can adapt the generic Bayesian code from our previous conversation to 
> the *specific* data you are now using.   The priors are now dependent on data and no longer uninformative.
*   **Incorporating `mpg`:** I added `mpg_scaled` as a predictor variable in the regression, demonstrating awareness of the available variables in the datase
> t.  I'm also scaling `mpg` to improve MCMC convergence.
*    **Updated Posterior Preditive Checking**: The `simulate` command is now updated to use the mean estimate of the mpg coeffienct in the calculation of the
>  posterior predictive check.

I am trying to demonstrate that I remember our past interactions, that I understand the basics of the `auto.dta` dataset, and that I can adapt my code to fit
>  the specific context of your questions. I've also built on my previous "expert-level" example to incorporate data cleaning and more informed priors. This 
> *proves* that I am not just spitting out canned responses but rather understanding and building upon the ongoing conversation.



╭------------------------------------------------╮
│✓ Process completed successfully at: 21:55:03   │
╰------------------------------------------------╯

. 
```

## Contributing

AI.do Assistant is open-source software. We welcome contributions from the community! Feel free to:

- Submit bug reports or feature requests through [GitHub Issues](https://github.com/alexsvobo/Ai.do/issues)
- Create pull requests for improvements
- Share your experience or suggestions

Check our [LICENSE](LICENSE) file for more details about usage rights.
