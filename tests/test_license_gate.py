import pytest

from vllm_ascend_simllm import (
    SimLLMHostContractDescriptor,
    SourceEvidence,
    SourceMigrationBlocked,
    SourceRoute,
    validate_source_evidence,
)


def test_unknown_license_is_rejected() -> None:
    evidence = SourceEvidence(
        route=SourceRoute.LICENSED_MIGRATION,
        copyright_holders=("upstream authors",),
    )
    with pytest.raises(SourceMigrationBlocked, match="reusable license"):
        validate_source_evidence(evidence)


def test_documented_license_route_is_accepted() -> None:
    evidence = SourceEvidence(
        route=SourceRoute.LICENSED_MIGRATION,
        copyright_holders=("copyright holder",),
        license_expression="Apache-2.0",
        permission_record="provenance/permission-record.md",
    )
    validate_source_evidence(evidence)


def test_clean_room_route_requires_independent_authors() -> None:
    evidence = SourceEvidence(
        route=SourceRoute.CLEAN_ROOM,
        copyright_holders=("behavior authors",),
        behavior_spec="HOST_CONTRACT.md",
    )
    with pytest.raises(SourceMigrationBlocked, match="independent"):
        validate_source_evidence(evidence)


def test_descriptor_remains_import_only() -> None:
    descriptor = SimLLMHostContractDescriptor()
    assert descriptor.activation_status == "import_only"
    assert len(descriptor.protocols) == 4
