"""Single-operator dashboard and version-one review API."""

from datetime import timedelta
import secrets
import sqlite3

from flask import Flask, Response, abort, jsonify, redirect, render_template, request, session
from werkzeug.exceptions import HTTPException

from .store import Conflict, Store


def create_app(db_path, token, secret_key, secure_cookie=False):
    if not isinstance(token, str) or len(token) < 32 or len(secret_key or "") < 32:
        raise ValueError("Provide separate operator token and session key of at least 32 characters")
    if token == secret_key:
        raise ValueError("Use separate credentials for operator access and session signing")
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=secret_key,
        MAX_CONTENT_LENGTH=16 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
        SESSION_COOKIE_SECURE=secure_cookie,
        PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
    )
    store = Store(db_path)

    def equal(first, second):
        return isinstance(first, str) and secrets.compare_digest(first.encode(), second.encode())

    def csrf():
        if "csrf" not in session:
            session["csrf"] = secrets.token_urlsafe(32)
        return session["csrf"]

    @app.before_request
    def protect():
        if request.endpoint in {"static", "health"}:
            return None
        if request.endpoint == "login":
            if request.method == "POST" and not equal(request.form.get("csrf"), csrf()):
                abort(403)
            return None
        if request.path.startswith("/api/") and request.headers.get("Authorization"):
            if equal(request.headers["Authorization"], "Bearer " + token):
                return None
            abort(401)
        if not session.get("authenticated"):
            if request.path.startswith("/api/"):
                abort(401)
            return redirect("/login", 303)
        if request.method == "POST":
            candidate = request.headers.get("X-CSRF-Token") or request.form.get("csrf")
            if not equal(candidate, csrf()):
                abort(403)
        return None

    @app.after_request
    def headers(result):
        result.headers.update(
            {
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "X-Frame-Options": "DENY",
                "Content-Security-Policy": "default-src 'self'; script-src 'none'; style-src 'self'; "
                "img-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
            }
        )
        return result

    @app.errorhandler(HTTPException)
    def http_error(exc):
        if request.path.startswith("/api/"):
            response = exc.get_response()
            response.data = app.json.dumps({"error": exc.description})
            response.content_type = "application/json"
            return response
        return exc

    @app.errorhandler(sqlite3.OperationalError)
    def database_error(exc):
        app.logger.exception("Database operation failed")
        return jsonify(error="Storage unavailable"), 503

    @app.get("/healthz")
    def health():
        return jsonify(status="ok")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = None
        if request.method == "POST":
            if equal(request.form.get("token"), token):
                session.clear()
                session.permanent = True
                session["authenticated"] = True
                csrf()
                return redirect("/", 303)
            error = "The operator token was not accepted."
        return render_template("login.html", csrf=csrf(), error=error), 401 if error else 200

    @app.post("/logout")
    def logout():
        session.clear()
        return redirect("/login", 303)

    def query_number(key, default=None, maximum=2**63 - 1):
        raw = request.args.get(key)
        if raw is None:
            return default
        if not raw.isascii() or not raw.isdigit() or len(raw) > 19 or not 1 <= int(raw) <= maximum:
            abort(400, description=f"Invalid {key}")
        return int(raw)

    def listing():
        status = request.args.get("status", "")
        if status not in ("", "pending", "issued", "dismissed"):
            abort(400, description="Unknown case status")
        rows, cursor = store.cases(status, query_number("before"), query_number("limit", 25, 100))
        return status, rows, cursor

    def get_case(case_id):
        case = store.get(case_id) if case_id <= 2**63 - 1 else None
        if case is None:
            abort(404, description="Case not found")
        return case

    def dashboard_view(selected_id=None, error=None):
        status, rows, cursor = listing()
        selected = get_case(selected_id) if selected_id else (rows[0] if rows else None)
        return render_template(
            "dashboard.html",
            selected=selected,
            rows=rows,
            next_before=cursor,
            status=status,
            counts=store.counts(),
            csrf=csrf(),
            error=error,
        )

    @app.get("/")
    def dashboard():
        return dashboard_view(query_number("selected"))

    @app.get("/api/cases")
    def cases():
        _, rows, cursor = listing()
        return jsonify(items=rows, next_before=cursor)

    @app.get("/api/cases/<int:case_id>")
    def case_detail(case_id):
        return jsonify(get_case(case_id))

    @app.get("/api/cases/<int:case_id>/evidence")
    def evidence_image(case_id):
        jpeg = store.jpeg(case_id) if case_id <= 2**63 - 1 else None
        if jpeg is None:
            abort(404, description="Evidence not found")
        return Response(jpeg, mimetype="image/jpeg")

    @app.post("/api/cases/<int:case_id>/review")
    @app.post("/cases/<int:case_id>/review")
    def review(case_id):
        api = request.path.startswith("/api/")
        data = request.get_json() if api else request.form
        if api and not isinstance(data, dict):
            abort(400, description="Expected a JSON object")
        get_case(case_id)
        try:
            store.review(case_id, data.get("action"), data.get("plate"), data.get("note"))
        except (ValueError, Conflict) as exc:
            code = 409 if isinstance(exc, Conflict) else 400
            if api:
                return jsonify(error=str(exc)), code
            return dashboard_view(case_id, str(exc)), code
        app.logger.info("Case %d reviewed (%s)", case_id, data["action"])
        return jsonify(get_case(case_id)) if api else redirect(f"/?selected={case_id}", 303)

    return app
