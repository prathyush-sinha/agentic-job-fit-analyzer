"""Seed lists of public job-board tokens to collect from.

Greenhouse and Lever both expose public, programmatic job-board APIs intended
for embedding a company's listings on its own site. We consume those rather than
scraping sites (LinkedIn/Indeed/etc.) whose terms prohibit it.

Tokens that 404 or error are skipped at runtime, so this list can be generous —
companies migrate ATSes over time. Add/remove freely.
"""

# https://boards-api.greenhouse.io/v1/boards/<token>/jobs?content=true
GREENHOUSE_TOKENS: list[str] = [
    "stripe", "gitlab", "robinhood", "coinbase", "databricks", "dropbox",
    "lyft", "pinterest", "reddit", "doordash", "instacart", "figma", "brex",
    "ramp", "plaid", "retool", "anduril", "samsara", "hashicorp", "confluent",
    "twilio", "asana", "gusto", "benchling", "affirm", "chime", "faire",
    "flexport", "niantic", "opendoor", "roblox", "discord", "cloudflare",
    "airtable", "gopuff", "nuro", "scaleai", "webflow", "verkada", "rippling",
    "vimeo", "wealthsimple", "thumbtack", "sofi", "cruise", "applovin",
    "snyk", "1password", "datadog", "celonis",
]

# https://api.lever.co/v0/postings/<token>?mode=json
LEVER_TOKENS: list[str] = [
    "leverdemo", "matchgroup", "voleon", "brightwheel", "kapwing",
    "ro", "fanduel", "swordhealth", "binance",
]
