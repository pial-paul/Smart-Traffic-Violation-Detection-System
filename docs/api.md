# Review API

Start the console with `trafficlab serve`. The default base URL is `http://127.0.0.1:8080`. Use `Authorization: Bearer <TRAFFIC_TOKEN>` for API clients. Browser sessions also work and require the session CSRF token on writes. Authenticated responses have `Cache-Control: no-store`. Ingestion is a local CLI operation on trusted recordings.

| Method | Path | Successful response |
| --- | --- | --- |
| GET | `/healthz` | 200, public process liveness, `{"status":"ok"}` |
| GET | `/api/cases` | 200, `items` and `next_before` |
| GET | `/api/cases/{id}` | 200, one case with provenance |
| GET | `/api/cases/{id}/evidence` | 200, `image/jpeg` containing approach and crossing panels |
| POST | `/api/cases/{id}/review` | 200, the updated case |

## Inspect a queue

```sh
curl -H "Authorization: Bearer $TRAFFIC_TOKEN" 'http://127.0.0.1:8080/api/cases?status=pending&limit=10'
```

Cases sort by descending database ID. Acceptable statuses are `pending`, `issued` and `dismissed`. Page size defaults to 25 and ranges from 1 to 100. If `next_before` is an integer, request the next page with `before=<that value>` and the same filters. A null cursor ends the list.

A case includes its `id`, `run_id`, `camera`, `kind`, `frame`, `approach_frame`, `track_id`, normalized bottom center `x`/`y`, `detector`, `label`, `signal` and `occurred_at`. Frame numbers start at zero. `occurred_at` is the UTC time of the observed confirmation frame. Track IDs are local to a recording. `plate` is nullable observed OCR. `provenance` carries the camera configuration, recording start, software version, detector selection and OCR setting. `source_sha256` and `evidence_sha256` identify the corresponding byte sequences.

## Record a decision

Send JSON containing `action` and a nonblank `note` of up to 500 characters. Issuance also requires a confirmed `plate` of 4–12 ASCII letters or digits. The server normalizes it to uppercase. Dismissal takes a reason and leaves the challan ID empty.

```sh
curl -X POST 'http://127.0.0.1:8080/api/cases/1/review' \
  -H "Authorization: Bearer $TRAFFIC_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"action":"issue","plate":"LAB317","note":"Both frames and the plate were checked on the synthetic clip."}'
```

After issuance, `status` is `issued`, `confirmed_plate` contains the reviewed text, `reviewed_at` contains the UTC decision time and `challan` contains a database-local reference such as `CH-00000001`. The original OCR remains available. A request with `{"action":"dismiss","note":"The vehicle belongs to an adjacent approach."}` instead records a dismissal.

## Error handling

API errors include an `error` string. Invalid input or a malformed cursor returns 400. Missing or incorrect credentials return 401, a failed browser CSRF check returns 403, and a missing case returns 404. Repeated or conflicting reviews return 409. Oversized request bodies return 413, a non-JSON review returns 415, and storage unavailability can return 503. If a write times out, read the case before retrying. Review and challan allocation share one transaction.
