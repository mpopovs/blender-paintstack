"""Builds the Principled BSDF network that samples the composited channel maps.
Only nodes named with the PS_ prefix are touched; anything else the user built
in the material is left alone."""

import bpy

from . import pixels
from .channels import CHANNELS, enabled_channels

NODE_PREFIX = "PS_"


def ensure_outputs(mat):
    ts = mat.paintstack
    for ch in enabled_channels(ts):
        if ts.output(ch.id) is not None:
            continue
        out = ts.outputs.add()
        out.channel = ch.id
        out.image = pixels.new_image(f"{mat.name}_{ch.label.replace(' ', '')}", ts.res, ch.default,
                                     float_buffer=ch.float_buffer, srgb=ch.srgb)


def _principled(nt):
    for n in nt.nodes:
        if n.type == "BSDF_PRINCIPLED":
            return n
    out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
    if out is None:
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        out.location = (300, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return bsdf


def _unlink(nt, socket):
    for link in list(socket.links):
        nt.links.remove(link)


def build(mat):
    ts = mat.paintstack
    if not ts.initialized:
        return
    if not mat.node_tree:
        mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        if n.name.startswith(NODE_PREFIX):
            nt.nodes.remove(n)
    bsdf = _principled(nt)
    x, y = bsdf.location.x - 600, bsdf.location.y + 300

    normal_out = None
    baked_normal = ts.baked_map("NORMAL")
    if ts.use_baked_normal and baked_normal is not None:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.name = tex.label = NODE_PREFIX + "Baked Normal"
        tex.image = baked_normal
        tex.location = (x - 300, y - 400)
        tex.hide = True
        nmap = nt.nodes.new("ShaderNodeNormalMap")
        nmap.name = NODE_PREFIX + "Normal Map"
        nmap.location = (x, y - 400)
        nt.links.new(tex.outputs["Color"], nmap.inputs["Color"])
        normal_out = nmap.outputs["Normal"]
        _unlink(nt, bsdf.inputs["Normal"])
        nt.links.new(normal_out, bsdf.inputs["Normal"])

    enabled = {c.id for c in enabled_channels(ts)}
    for ch in CHANNELS:
        img = ts.output(ch.id)
        if ch.id not in enabled or img is None:
            continue
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.name = tex.label = NODE_PREFIX + ch.label
        tex.image = img
        tex.location = (x, y)
        tex.hide = True
        y -= 60

        if ch.id == "HEIGHT":
            bump = nt.nodes.new("ShaderNodeBump")
            bump.name = NODE_PREFIX + "Bump"
            bump.location = (x + 300, y)
            bump.inputs["Strength"].default_value = ts.bump_strength
            nt.links.new(tex.outputs["Color"], bump.inputs["Height"])
            if normal_out is not None:
                nt.links.new(normal_out, bump.inputs["Normal"])
            _unlink(nt, bsdf.inputs["Normal"])
            nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
            continue

        sock = bsdf.inputs.get(ch.socket)
        if sock is None:
            continue
        _unlink(nt, sock)
        nt.links.new(tex.outputs["Color"], sock)
        if ch.id == "EMISSION" and "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = 1.0
