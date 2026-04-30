"""Unsupported capability response helpers."""

from __future__ import annotations

from typing import Any

from tradingdev.app.feature_request_service import FeatureRequestService


class CapabilityService:
    """Create structured unsupported responses and feature requests."""

    def __init__(self, feature_requests: FeatureRequestService | None = None) -> None:
        self._feature_requests = feature_requests or FeatureRequestService()

    def unsupported(
        self,
        *,
        tool: str,
        reason: str,
        feature_title: str,
        feature_description: str,
    ) -> dict[str, Any]:
        """Return an unsupported response and record the request."""
        request = self._feature_requests.record(
            title=feature_title,
            description=feature_description,
            source_tool=tool,
        )
        return {
            "success": False,
            "unsupported": True,
            "tool": tool,
            "reason": reason,
            "feature_request": request,
        }
