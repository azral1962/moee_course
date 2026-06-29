# MOEE Agentic Simulation

This package is a small, object-oriented reference implementation of the paper's MOEE/TISE model. It separates:

- stakeholder state and VOMR accounting;
- stakeholder MIG reasoning (historical, probabilistic, agentic, or hybrid);
- MIG setup (independent, prenegotiated smart contract, or adaptive PUDAL);
- PSKVE resource-to-output execution; and
- optional LLM providers (offline heuristic, Ollama, OpenAI, or DeepSeek).

The default run is offline and deterministic—no API key or external package is required.

## Quick start

From the repository root:

```powershell
python -m simulation.cli --architecture baseline --reasoning historical
python -m simulation.cli --architecture sota --reasoning probabilistic
python -m simulation.cli --architecture tise --reasoning agent --provider heuristic
```

Write step-level results to CSV:

```powershell
python -m simulation.cli --architecture tise --reasoning hybrid --steps 24 --output results/moee_run.csv
```

## Provider configuration

Copy `simulation/.env.example` to `.env` in the repository root or `simulation/.env`, fill in only the provider you plan to use, and never commit real keys.

### Ollama (local)

Start Ollama and make sure the configured model is installed, then run:

```powershell
python -m simulation.cli --architecture tise --reasoning agent --provider ollama
```

The local adapter calls `http://localhost:11434/api/chat` by default.

### OpenAI

Set `OPENAI_API_KEY` and optionally `OPENAI_MODEL`, then run with `--provider openai`. The adapter uses the Responses API and requests a JSON-only decision.

### DeepSeek

Set `DEEPSEEK_API_KEY` and optionally `DEEPSEEK_MODEL`, then run with `--provider deepseek`. The adapter uses the provider's OpenAI-compatible chat-completions endpoint.

All remote-provider failures raise `ProviderError` by default. Pass `--allow-agent-fallback` to fall back to the auditable local heuristic during a simulation run.

## Tests

```powershell
python -m unittest discover -s simulation/tests -v
```

The tests never call external services; fake and heuristic providers exercise the agentic paths.

## Package map

- `models.py` — stakeholders, missions, MIG proposals, VOMR, and scenarios
- `reasoning.py` — stakeholder MIG reasoners
- `providers.py` — Ollama, OpenAI, DeepSeek, fake, and heuristic adapters
- `coordination.py` — independent, smart-contract, and PUDAL coordination
- `markets.py` — Stock, Financial, and Resource–Output market signals
- `exchange.py` — PSKVE execution and accounting
- `engine.py` — reusable simulation loop and demo factory
- `cli.py` — command-line runner

Provider documentation used for the adapters:

- OpenAI text generation and Responses API: <https://developers.openai.com/api/docs/guides/text>
- Ollama chat API: <https://docs.ollama.com/api/chat>
- DeepSeek chat completion API: <https://api-docs.deepseek.com/api/create-chat-completion>
