# Repository Guidelines

## Project Structure

This is one repository with a Python backend and Vue frontend. Run Git commands at the root.

- `backend/medagent/`: API, contracts, risk screening, classification, retrieval, memory, execution, monitoring and evaluation.
- `backend/resources/`: reviewed public reference summaries, role Skills and synthetic evaluation cases.
- `backend/tests/`: deterministic pytest tests; CI runs them with the network guard (`MAR_OFFLINE_TESTS=1`). `backend/artifacts/` holds synthetic validation reports locally and is ignored by Git.
- `frontend/src/`: workbench, API client and CSS; `frontend/tests/` contains Node tests.
- `docs/`: local-only and ignored by Git: `guides/` (handover, demo, operations), `design/` (module design), `plans/` (plans and experiment protocols) and `records/` (dated acceptance and repair records by scenario); see `docs/README.md`. `.local-backup/` and `local-notes/` are ignored too.

## Development Commands

From `backend/`, install with `python -m pip install -r requirements-dev.txt`, run `python -m pytest tests/ -q`, and start `python -m uvicorn medagent.app:app --reload --port 8000`. Copy `.env.example` to ignored `.env.local` and configure your model and storage first. Developer routes and workbench views need `MAR_DEVELOPER_TOOLS=true`; the default is user mode, where they answer 404.

From the root, `docker compose --env-file backend/.env.local up -d redis chroma` starts storage. Add `--build` without service names to build the full stack.

From `frontend/`, use `npm ci`, `npm test`, `npm run dev`, and `npm run build`. Set `VITE_API_TARGET` to override the development API proxy.

## Coding Conventions

Use four spaces, type hints on public Python interfaces, `snake_case` functions and `PascalCase` classes. JavaScript and Vue use two spaces, single quotes, and no semicolons. Keep HTTP access in `client.js`; render answers as text and validate reference links. Keep role permissions and risk checks outside model prompts.

## Testing and Evaluation

Name tests `test_*.py` and `test_*`; use `asyncio.run` for asynchronous cases. Unit tests must not call paid models or download embedding weights. Cover changed failure behavior, scope isolation and output validation. Frontend changes need client tests when behavior changes and a production build. No coverage threshold is configured.

Run `python -m medagent.evaluate` only for explicitly authorized real model validation. Preserve Judge errors, unknown Token usage and failed cases. Never carry scores between implementations or claim clinical validation.

`python -m medagent.comparison` and `python -m medagent.benchmark` save a plan without model calls; `--execute` needs explicit online scope and budget. Keep paired snapshots, cold/warm cache results, failures and synthetic counts distinct from real provider measurements.

## Commits and Configuration

Use focused, descriptive commit messages. PR descriptions explain behavior, validation and limitations; include UI screenshots when relevant. Never commit `.env.local`, credentials, raw consultations, databases, dependency folders or recovery snapshots. Maintain `THIRD_PARTY.md` when adding external code or reference material.
