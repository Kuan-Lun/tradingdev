"""HTTP capability delivery for registered reading artifacts."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode, urlsplit

from starlette.concurrency import run_in_threadpool
from starlette.responses import Response

from tradingdev.app.contracts.common import ErrorResponse
from tradingdev.app.contracts.research import ArtifactDownload

if TYPE_CHECKING:
    from mcp.server.mcpserver import Context
    from starlette.requests import Request

    from tradingdev.app.artifact_service import ArtifactService

DOWNLOAD_TTL_SECONDS = 900
DOWNLOAD_ROUTE = "/artifact-download/{artifact_id}"
_HEADERS = {
    "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}
# Keep offline report scripts/Blob downloads, but give the document an opaque
# origin; block network/API access, forms and popups.
_HTML_CSP = (
    "sandbox allow-scripts allow-downloads; default-src 'none'; "
    "script-src 'unsafe-inline' 'unsafe-eval' blob:; style-src 'unsafe-inline'; "
    "img-src data: blob:; font-src data:; frame-src blob: data:; "
    "connect-src 'none'; form-action 'none'; base-uri 'none'; frame-ancestors 'none'"
)


def validate_public_base_url(value: str | None) -> str | None:
    """Accept an explicit public HTTP origin and optional unencoded proxy prefix."""
    if value is None:
        return None
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or "?" in value
        or "#" in value
        or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value)
        or "\\" in value
        or "*" in value
        or (parsed.port is not None and not 1 <= parsed.port <= 65535)
        or any(
            part in {".", ".."} or re.fullmatch(r"[A-Za-z0-9._~-]+", part) is None
            for part in parsed.path.strip("/").split("/")
            if parsed.path.strip("/")
        )
        or "//" in parsed.path
    ):
        raise ValueError(
            "Public base URL must be http(s)://host[:port][/proxy-prefix] "
            "without credentials, query or fragment"
        )
    return value.rstrip("/")


class ArtifactDownloads:
    """Custom routes bypass MCP auth: each URL is a short-lived bearer capability.

    The instance key is never persisted. Restarting invalidates URLs, not artifacts.
    Both issuance and GET verify registration and bytes; neither executes research.
    """

    def __init__(
        self, service: ArtifactService, *, public_base_url: str | None = None
    ) -> None:
        self.service = service
        self.public_base_url = validate_public_base_url(public_base_url)
        self._key = secrets.token_bytes(32)

    def _signature(self, artifact_id: str, sha256: str, expires: int) -> str:
        message = f"{artifact_id}:{sha256}:{expires}".encode()
        return hmac.new(self._key, message, hashlib.sha256).hexdigest()

    def issue(self, artifact_id: str, sha256: str, base_url: str) -> ArtifactDownload:
        artifact = self.service.read_download(artifact_id, sha256)
        expires = int(time.time()) + DOWNLOAD_TTL_SECONDS
        query = urlencode(
            {
                "sha256": sha256,
                "expires": expires,
                "signature": self._signature(artifact_id, sha256, expires),
            }
        )
        base = self.public_base_url or base_url.rstrip("/")
        return ArtifactDownload(
            artifact_id=artifact_id,
            sha256=sha256,
            mime_type=artifact.mime_type,
            download_url=(
                f"{base}/artifact-download/{quote(artifact_id, safe='')}?{query}"
            ),
            download_expires_at=expires,
        )

    def for_context(
        self, artifact_id: str, sha256: str, ctx: Context[Any, Any]
    ) -> ArtifactDownload | ErrorResponse:
        try:
            request = ctx.request_context.request
        except ValueError:
            request = None
        if request is None:
            return ErrorResponse(
                success=False,
                error=(
                    "HTTP delivery is unavailable over stdio; use verified client "
                    "file delivery or get_artifact text pages."
                ),
                code="artifact_delivery_unavailable",
            )
        try:
            return self.issue(artifact_id, sha256, str(request.base_url))
        except ValueError as error:
            return ErrorResponse(
                success=False,
                error=f"{error}. No file was regenerated.",
                code="artifact_delivery_invalid",
            )
        except OSError:
            return ErrorResponse(
                success=False,
                error=(
                    "Saved artifact could not be read; inspect its saved identity "
                    "and file availability. No file was regenerated."
                ),
                code="artifact_delivery_invalid",
            )

    async def serve(self, request: Request) -> Response:
        artifact_id = request.path_params["artifact_id"]
        sha256 = request.query_params.get("sha256", "")
        signature = request.query_params.get("signature", "")
        try:
            expires = int(request.query_params.get("expires", "0"))
        except ValueError:
            expires = 0
        if (
            re.fullmatch(r"[0-9a-f]{64}", signature) is None
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
            or expires <= time.time()
            or not hmac.compare_digest(
                signature, self._signature(artifact_id, sha256, expires)
            )
        ):
            return Response(
                "Invalid or expired download link", status_code=403, headers=_HEADERS
            )
        try:
            artifact = await run_in_threadpool(
                self.service.read_download, artifact_id, sha256
            )
        except ValueError, OSError:
            return Response(
                "Saved artifact unavailable; inspect its saved identity",
                status_code=404,
                headers=_HEADERS,
            )
        html = artifact.mime_type == "text/html"
        disposition = "inline" if html else "attachment"
        return Response(
            artifact.content,
            media_type=artifact.mime_type,
            headers={
                **_HEADERS,
                "Content-Disposition": f'{disposition}; filename="{artifact.filename}"',
                **({"Content-Security-Policy": _HTML_CSP} if html else {}),
            },
        )
