"""Fixed, non-executable program identity for manifest/storage/presentation tests."""

from tradingdev.domain.runtime import RuntimeSpec
from tradingdev.domain.strategies.execution import GeneratedProgram

GENERATED_PROGRAM = GeneratedProgram(
    source_hash="b" * 64,
    class_name="FixtureStrategy",
    runtime=RuntimeSpec(image_id="sha256:" + "a" * 64),
)
