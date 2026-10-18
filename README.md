# Forge Front-Desk Agent

Safety-first front-desk verification and emergency-routing flow built with FastAPI, LangGraph, SQLite, Pydantic, YAML, and an OpenRouter-compatible model factory.

## Run

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Add OPENROUTER_API_KEY to .env when ready
uvicorn forge_agent.api:app --reload
```

Post `messages_sample.json`-shaped JSON to `http://127.0.0.1:8000/process`. The service also exposes `GET /health`.

Without an API key, deterministic classification is used so tests and local runs work safely. With a key, OpenRouter is used only for structured intent/urgency and medication extraction; verification, matching, policy, routing, and emergency handling remain deterministic.

Supported verified workflows fan out concurrently by intent. Clinical, refill, billing, schedule, and reschedule branches return isolated results; the graph merges them into one state, then writes the merged result and complete state to `executions.db`. Scheduling checks clinic hours, 20-minute slot boundaries, lunch, holidays, provider leave, and overlapping appointments. LangGraph checkpoints are stored separately in `checkpoints.db`.

Run the sample evaluation later with:

```powershell
python -m evals.automatic_test_runner
# Optional: change or disable the delay
python -m evals.automatic_test_runner --delay-seconds 10
python -m evals.automatic_test_runner --delay-seconds 0
```

Each joined message/label result is appended immediately to `evals/post_runs.jsonl`.

To run the two batches separately with JSON output files:

```powershell
python -m evals.automatic_test_runner --start-entry 1 --end-entry 20 --output evals/output_results_1_to_20.json
python -m evals.automatic_test_runner --start-entry 21 --end-entry 40 --output evals/output_results_21_to_40.json
```

For one file per completed message, use:

```powershell
python -m evals.automatic_test_runner --start-entry 1 --end-entry 20 --results-dir evals/results
```

This writes `evals/results/message1.json` through `message20.json` immediately after each request. Run the second batch with `--start-entry 21 --end-entry 40`; existing files are not deleted.

Calculate final metrics from saved results:

```powershell
python -m evals.final_metrics
```

This writes `evals/final_metrics.json` with urgency accuracy, exact action accuracy, action precision/recall, emergency recall, missed emergencies, and `must_not` violations. The newest saved result for each message ID is used.
