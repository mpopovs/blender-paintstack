"""PBR channel definitions shared by every part of the add-on."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Channel:
    id: str
    label: str
    is_color: bool
    default: tuple  # RGB, stored values (sRGB-encoded for sRGB channels)
    srgb: bool
    float_buffer: bool
    socket: str  # Principled BSDF input, or "" when wired specially

    @property
    def key(self):
        return self.id.lower()


CHANNELS = (
    Channel("BASE_COLOR", "Base Color", True, (0.5, 0.5, 0.5), True, False, "Base Color"),
    Channel("METALLIC", "Metallic", False, (0.0, 0.0, 0.0), False, False, "Metallic"),
    Channel("ROUGHNESS", "Roughness", False, (0.5, 0.5, 0.5), False, False, "Roughness"),
    Channel("HEIGHT", "Height", False, (0.5, 0.5, 0.5), False, True, ""),
    Channel("EMISSION", "Emission", True, (0.0, 0.0, 0.0), True, False, "Emission Color"),
)

BY_ID = {c.id: c for c in CHANNELS}

CHANNEL_ITEMS = [(c.id, c.label, "") for c in CHANNELS]


def enabled_channels(ts):
    """Channels switched on for a texture set, in canonical order."""
    return [c for c in CHANNELS if getattr(ts, "use_" + c.key)]
