# Drainage database dump

`drainage_dump.sql.gz` is a full `pg_dump` of the local PostGIS database
that `web_app/geoai_agent/drainage_db.py` reads from (stormwater pipes and catch
basins for DC, Austin, Charlotte, Anne Arundel, Fairfax, Loudoun, and PG
County -- loaded via the `web_app/load_*_stormwater.py` scripts).

## Restoring it

1. Create an empty PostGIS-enabled database and point `DATABASE_URL` at it
   (see the root `README`/`.env` for the expected format).
2. Restore:

   ```bash
   gunzip -c db_dump/drainage_dump.sql.gz | psql "$DATABASE_URL"
   ```

The dump includes the `postgis` extension objects, so the target database
just needs the `postgis` extension available (not necessarily already
created) before restoring.
