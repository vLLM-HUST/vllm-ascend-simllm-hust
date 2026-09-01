"""Fail-closed source migration gate for the SimLLM extension.

This module contains no SimLLM implementation. It records the evidence that a
maintainer must supply before historical implementation code can be imported
into this Apache-2.0 repository.
"""

from dataclasses import dataclass
from enum import StrEnum


class SourceRoute(StrEnum):
    """Permitted routes for supplying a runtime implementation."""

    LICENSED_MIGRATION = "licensed_migration"
    CLEAN_ROOM = "clean_room"


@dataclass(frozen=True)
class SourceEvidence:
    """Reviewable evidence for one implementation source route."""

    route: SourceRoute
    copyright_holders: tuple[str, ...]
    license_expression: str | None = None
    permission_record: str | None = None
    behavior_spec: str | None = None
    independent_authors: tuple[str, ...] = ()


class SourceMigrationBlocked(RuntimeError):
    """Raised when source provenance is not sufficient for migration."""


def validate_source_evidence(evidence: SourceEvidence) -> None:
    """Accept only a documented licensed migration or clean-room route."""

    if not evidence.copyright_holders:
        raise SourceMigrationBlocked("copyright holders must be identified")

    if evidence.route is SourceRoute.LICENSED_MIGRATION:
        if not evidence.license_expression:
            raise SourceMigrationBlocked("a reusable license must be declared")
        if not evidence.permission_record:
            raise SourceMigrationBlocked("the license or permission record is required")
        return

    if evidence.route is SourceRoute.CLEAN_ROOM:
        if not evidence.behavior_spec:
            raise SourceMigrationBlocked(
                "a reviewable behavior specification is required"
            )
        if not evidence.independent_authors:
            raise SourceMigrationBlocked(
                "independent implementation authors are required"
            )
        return

    raise SourceMigrationBlocked(f"unsupported source route: {evidence.route}")


@dataclass(frozen=True)
class SimLLMHostContractDescriptor:
    """Non-activating descriptor for the host seams needed by SimLLM."""

    protocols: tuple[str, ...] = (
        "vllm.scheduler.rewrite-policy.v1",
        "vllm.kv-reuse.index.v1",
        "vllm.embedding.provider.v1",
        "vllm.model-runner.slot-mapping.v1",
    )
    activation_status: str = "import_only"
