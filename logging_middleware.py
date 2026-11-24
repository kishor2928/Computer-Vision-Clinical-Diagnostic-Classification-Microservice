"""
Simple structured request logging + basic exception handling
for the Flask microservice.
"""

import time
import uuid
import logging
from flask import request, g, jsonify
from functools import wraps

logger = logging.getLogger("diagnostic_service")


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def request_id_middleware():
    """Assign a request id early so every log line can reference it."""
    g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    g.start_time = time.perf_counter()


def log_request_end(response):
    duration_ms = (time.perf_counter() - getattr(g, "start_time", time.perf_counter())) * 1000
    logger.info(
        "request_finished",
        extra={
            "request_id": getattr(g, "request_id", "-"),
            "method": request.method,
            "path": request.path,
            "status": response.status_code,
            "duration_ms": round(duration_ms, 1),
            "remote_addr": request.remote_addr,
        },
    )
    # also put the id in the response header for clients
    response.headers["X-Request-ID"] = getattr(g, "request_id", "-")
    return response


def handle_exceptions(app):
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": "bad_request", "message": str(e)}), 400

    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"error": "payload_too_large", "message": "Image exceeds size limit"}), 413

    @app.errorhandler(500)
    def internal(e):
        logger.exception("unhandled_exception")
        return jsonify({"error": "internal_error", "message": "Something went wrong"}), 500

    @app.errorhandler(Exception)
    def catch_all(e):
        logger.exception("unhandled_exception", extra={"request_id": getattr(g, "request_id", "-")})
        return jsonify({"error": "internal_error", "message": str(e)}), 500
