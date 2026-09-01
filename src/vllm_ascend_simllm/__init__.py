"""vllm-ascend-simllm inert contract descriptor."""

from .license_gate import (
    SimLLMHostContractDescriptor,
    SourceEvidence,
    SourceMigrationBlocked,
    SourceRoute,
    validate_source_evidence,
)

__all__ = [
    "SimLLMHostContractDescriptor",
    "SourceEvidence",
    "SourceMigrationBlocked",
    "SourceRoute",
    "validate_source_evidence",
]
