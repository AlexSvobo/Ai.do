{smcl}
{* *! version 1.0.0 23feb2025}{...}
{viewerjumpto "Syntax" "aido##syntax"}{...}
{viewerjumpto "Description" "aido##description"}{...}
{viewerjumpto "Options" "aido##options"}{...}
{viewerjumpto "Examples" "aido##examples"}{...}
{title:AI.do Assistant}

{marker syntax}{...}
{title:Syntax}

{p 8 17 2}
{cmdab:aido_config},
{opt prov:ider(provider)} 
{opt apik:ey(string)} 
[{opt mod:elid(string)}]

{p 8 17 2}
{cmd:aido} "{it:query_text}"

{synoptset 20 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Configuration}
{synopt:{opt prov:ider(string)}}AI provider (huggingface, gemini){p_end}
{synopt:{opt apik:ey(string)}}API key from provider{p_end}
{synopt:{opt mod:elid(string)}}Model ID (required for huggingface, e.g. "Qwen/Qwen2.5-Coder-32B"){p_end}
{synoptline}

{marker description}{...}
{title:Description}
{pstd}
{cmd:aido} provides an AI-powered assistant for Stata, helping with data analysis, 
visualization, and statistical programming tasks. The assistant maintains context of:

{pmore}- Your current dataset (variables, types, labels, summary statistics){p_end}
{pmore}- Recent estimation results (regression output, test statistics){p_end}
{pmore}- Command history{p_end}
{pmore}- Previous chat messages{p_end}

{pstd}
Queries must be enclosed in quotes. You must at least start the quote.
For multi-word queries, use: {cmd:aido "your query here...}

{marker examples}{...}
{title:Examples}

    {hline}
    Initial Setup
    {hline}
    
    {cmd:. * For Gemini API}
    {cmd:. aido_config, provider(gemini) apikey("YOUR-GEMINI-KEY")}
    
    {cmd:. * For Hugging Face}
    {cmd:. aido_config, provider(huggingface) apikey("YOUR-HF-KEY") modelid("Qwen/Qwen2.5-Coder-32B")}

    {hline}
    Basic Usage
    {hline}
    
    {cmd:. sysuse auto}
    {cmd:. aido "Describe the relationship between price and mpg"}
    {cmd:. aido "Create a histogram of weights by foreign vs domestic"}
    {cmd:. aido "Run a regression of price on mpg and weight"}
    {cmd:. aido "Interpret these results"}
    {cmd:. aido "Test for heteroskedasticity"}

    {hline}
    Advanced Analysis
    {hline}
    
    {cmd:. aido "Find outliers in the price variable"}
    {cmd:. aido "Create dummy variables for repair categories"}
    {cmd:. aido "What's the correlation between engine size and fuel efficiency?"}
    {cmd:. aido "Generate interaction terms for price and weight"}
    {cmd:. aido "Explain the distribution of repair records"}

{marker author}{...}
{title:Author}

{pstd}
Alex Svoboda{break}
Website: {browse "https://github.com/AlexSvobo":GitHub Profile}

{marker contact}{...}
{title:Contact}

{pstd}
For bug reports and feature requests:{break}
{browse "https://github.com/AlexSvobo/Ai.do/issues":GitHub Issues}

{marker license}{...}
{title:License}

{pstd}
MIT License. See {browse "https://github.com/AlexSvobo/Ai.do/blob/main/LICENSE":LICENSE} for details.

{hline}