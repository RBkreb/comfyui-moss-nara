from .nodes import MossNaraExtension


async def comfy_entrypoint() -> MossNaraExtension:
    return MossNaraExtension()