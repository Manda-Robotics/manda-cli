# Hybrid dashboard architecture

## Storage roles

Backblaze B2 is the canonical source of truth for immutable experiment outputs. Supabase Postgres
is a rebuildable read model for low-latency dashboard discovery, filtering, aggregation, access
control, and future annotations.

The database stores no MP4 or HDF5 bodies. Its `assets` rows contain bucket/object keys and verified
manifest metadata. A future authenticated signing endpoint will exchange an authorized asset row
for a short-lived B2 URL; browsers will stream media directly from B2.

## Database contract

The migration at `supabase/migrations/20260823000100_dashboard_read_model.sql` creates:

- `studies`: top-level benchmark identity;
- `runs`: immutable run identity, manifest digest, lifecycle timestamps, and indexing state;
- `task_results`: precomputed task-level summaries;
- `episodes`: filterable episode data plus the complete source row in `raw`;
- `assets`: B2 locations and manifest metadata for every run object; and
- `catalog_imports`: synchronization receipts and counts.

All tables have Row Level Security enabled. `anon` receives no access. Authenticated users may read
only runs whose indexing status is `ready`; the server-side Supabase secret maps to `service_role`
for ingestion and bypasses RLS. Never expose `SUPABASE_SECRET_KEY` to frontend code.

## Apply the migration

The Supabase project URL and Data API keys cannot execute arbitrary schema SQL. Apply migrations
with an authenticated Supabase CLI session:

```bash
cd /path/to/manda-cli
supabase login
supabase link --project-ref YOUR_PROJECT_REF
supabase db push
```

For the currently configured project, the project reference is the hostname prefix from
`SUPABASE_URL`. Do not commit CLI access tokens, database passwords, or populated environment
files.

## Synchronize committed B2 runs

Load both private configurations. `dashboard doctor` is read-only, and `dashboard sync` is dry-run
unless `--yes` is present:

```bash
set -a
source /path/to/storage.env
source /path/to/supabase.env
set +a

manda dashboard doctor
manda dashboard sync \
  --study-id state-of-general-robot-policies-2026 \
  --catalog-version initial-20260823

manda dashboard sync \
  --study-id state-of-general-robot-policies-2026 \
  --catalog-version initial-20260823 \
  --yes
```

The synchronizer lists only committed B2 runs, reads their result rows and manifests, and builds
the relational projection. For each changed run it:

1. writes the run with `index_status = 'indexing'`, which keeps it hidden from authenticated reads;
2. replaces that run's task, episode, and asset children;
3. marks the run `ready`; and
4. writes a `catalog_imports` receipt after all runs finish.

Runs already indexed with the same manifest and index-schema version are skipped. Runs absent from
a filtered catalog are not deleted. If ingestion fails, the affected run is marked `failed` and a
later retry can rebuild it.

## Next web phase

The next increment should add:

1. an authenticated media-signing endpoint that reads `assets`, authorizes the user, and produces a
   short-lived B2 `GetObject` URL;
2. a database-backed dashboard loader for run/task/episode queries;
3. B2 CORS for the deployed dashboard origin and HTTP range requests; and
4. precomputed trajectory chart payloads so web requests never remotely open raw HDF5 files.
