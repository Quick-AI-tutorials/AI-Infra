# OpenClaw Telegram Setup

```
openclaw agents list --bindings

Agents:
- main (default)
  Routing rules: 0
  Routing: default (no explicit rules)
- taxagent (TaxAgent)
  Routing rules: 0

openclaw agents bind --agent taxagent --bind telegram:tax_agent_jessup

Go to Telegram and Type Hi to that agent -> It will generate a pairing code BT9ACWXX

# Then, in the terminal
openclaw pairing approve telegram BT9ACWXX 
```
