from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import tempfile
import unittest

from trafficlab.store import Conflict, Store
from trafficlab.web import create_app


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        path = Path(self.directory.name) / "store.db"
        self.store = Store(path)
        self.token = "a" * 42
        self.app = create_app(path, self.token, "b" * 42)
        self.app.testing = True
        self.client = self.app.test_client()
        self.auth = {"Authorization": "Bearer " + self.token}
        self.event = dict(
            frame=12,
            track_id=2,
            camera="lab",
            detector="motion",
            plate=None,
            occurred_at="2025-01-01T00:00:01.200+00:00",
        )
        self.jpeg = b"fixture bytes"
        self.store.save("lab", "synthetic", {}, {}, [(self.event, self.jpeg)])

    def test_anonymous_routes_hide_cases_and_evidence(self):
        self.assertEqual(self.client.get("/").status_code, 303)
        for url in ("/api/cases", "/api/cases/1", "/api/cases/1/evidence"):
            self.assertEqual(self.client.get(url).status_code, 401)
        self.assertEqual(self.client.get("/healthz").json, {"status": "ok"})

    def test_case_filter_metadata_and_evidence_digest(self):
        items = self.client.get("/api/cases?status=pending", headers=self.auth).json["items"]
        body = self.client.get("/api/cases/1/evidence", headers=self.auth)
        self.assertEqual(hashlib.sha256(body.data).hexdigest(), items[0]["evidence_sha256"])
        self.assertEqual(body.headers["Cache-Control"], "no-store")
        for query in ("status=typo", "before=-1", "limit=101", "limit="):
            self.assertEqual(self.client.get("/api/cases?" + query, headers=self.auth).status_code, 400)
        self.assertEqual(
            self.client.get("/api/cases/9999999999999999999999", headers=self.auth).status_code, 404
        )

    def test_cursor_pages_without_duplicates(self):
        self.store.save("lab", "other", {}, {}, [(self.event, self.jpeg)])
        page = self.client.get("/api/cases?limit=1", headers=self.auth).json
        self.assertEqual([v["id"] for v in page["items"]], [2])
        next_page = self.client.get(
            f"/api/cases?limit=1&before={page['next_before']}", headers=self.auth
        ).json
        self.assertEqual([v["id"] for v in next_page["items"]], [1])
        self.assertIsNone(next_page["next_before"])

    def test_token_review_validation_and_conflict(self):
        url = "/api/cases/1/review"
        data = {"action": "issue", "note": "Plate verified", "plate": "AB12CD"}
        for invalid in (
            {**data, "action": "unknown"},
            {**data, "plate": None},
            {**data, "note": "  "},
            [],
        ):
            self.assertEqual(self.client.post(url, json=invalid, headers=self.auth).status_code, 400)
        response = self.client.post(url, json=data, headers=self.auth)
        self.assertEqual(response.json["challan"], "CH-00000001")
        self.assertIsNone(response.json["plate"])
        self.assertEqual(response.json["confirmed_plate"], "AB12CD")
        self.assertEqual(self.client.post(url, json=data, headers=self.auth).status_code, 409)

    def test_browser_csrf_and_signout(self):
        self.client.get("/login")
        self.assertEqual(self.client.post("/login", data={"token": self.token}).status_code, 403)
        with self.client.session_transaction() as session:
            first = session["csrf"]
        response = self.client.post("/login", data={"token": self.token, "csrf": first})
        self.assertEqual(response.status_code, 303)
        with self.client.session_transaction() as session:
            fresh = session["csrf"]
        self.assertNotEqual(first, fresh)
        self.assertEqual(self.client.get("/").status_code, 200)
        data = dict(action="dismiss", note="Signal ambiguous", csrf=first)
        url = "/cases/1/review"
        self.assertEqual(self.client.post(url, data=data).status_code, 403)
        self.assertEqual(self.client.post(url, data={**data, "csrf": fresh}).status_code, 303)
        self.assertIsNone(self.store.get(1)["challan"])
        self.client.post("/logout", data={"csrf": fresh})
        self.assertEqual(self.client.get("/api/cases").status_code, 401)

    def test_concurrent_review_has_one_winner(self):
        def decide(action):
            try:
                self.store.review(1, action, "AB12CD", "Reviewed")
                return "ok"
            except Conflict:
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertCountEqual(pool.map(decide, ["issue", "dismiss"]), ["ok", "conflict"])

    def test_partial_ingestion_rolls_back_all_records(self):
        def broken():
            yield self.event, self.jpeg
            raise OSError("Interrupted")

        with self.assertRaises(OSError):
            self.store.save("lab", "interrupted", {}, {}, broken())
        self.assertIsNone(self.store.existing("lab", "interrupted"))
        self.assertEqual(self.store.counts()["pending"], 1)
