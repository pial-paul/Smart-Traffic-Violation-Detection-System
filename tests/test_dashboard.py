"""Check review affordances that differ between pending and closed cases."""

import unittest

from flask import Flask, render_template


class DashboardTemplateTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask("trafficlab")
        self.case = {
            "id": 7,
            "camera": "North <approach>",
            "status": "pending",
            "occurred_at": "2025-01-01T12:00:05+00:00",
            "plate": None,
            "track_id": 3,
            "frame": 60,
            "detector": "motion",
            "source_sha256": "a" * 64,
            "evidence_sha256": "b" * 64,
        }

    def render(self):
        with self.app.test_request_context("/?before=9"):
            return render_template(
                "dashboard.html",
                csrf="test-csrf",
                rows=[self.case],
                selected=self.case,
                status=self.case["status"],
                counts={self.case["status"]: 1},
                next_before=7,
            )

    def test_pending_case_keeps_filter_and_cursor_and_escapes_metadata(self):
        html = self.render()
        self.assertIn("selected=7&amp;status=pending&amp;before=9#case-detail", html)
        self.assertIn('src="/api/cases/7/evidence"', html)
        self.assertIn('action="/cases/7/review"', html)
        self.assertIn("North &lt;approach&gt;", html)
        self.assertIn("Unreadable", html)
        self.assertNotIn("North <approach>", html)

    def test_closed_case_displays_audit_record_without_review_form(self):
        self.case.update(
            status="issued",
            reviewed_at="2025-01-01T12:10:00+00:00",
            confirmed_plate="LAB317",
            challan="TEST-7",
            note="Reviewed both frames.",
        )
        html = self.render()
        self.assertIn("Recorded decision", html)
        self.assertIn("TEST-7", html)
        self.assertIn("LAB317", html)
        self.assertIn("Reviewed both frames.", html)
        self.assertNotIn('action="/cases/7/review"', html)
