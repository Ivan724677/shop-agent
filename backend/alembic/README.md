# Stage-seven database migrations

Run from the repository root after installing `backend/requirements.txt`:

```bash
alembic -c backend/alembic.ini upgrade head
```

The initial migration creates durable conversation, Agent run/checkpoint,
event, pending-action, handoff, feedback and RAG index-version tables.

