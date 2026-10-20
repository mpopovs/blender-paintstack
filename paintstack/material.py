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


def _multiply_ao(nt, color_out, ao_image, location):
    ao = nt.nodes.new("ShaderNodeTexImage")
    ao.name = ao.label = NODE_PREFIX + "Baked AO"
    ao.image = ao_image
    ao.location = (location[0] - 300, location[1] - 60)
    ao.hide = True
    mix = nt.nodes.new("ShaderNodeMix")
    mix.name = NODE_PREFIX + "AO Multiply"
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.location = location
    sock = {s.identifier: s for s in mix.inputs}
    sock["Factor_Float"].default_value = 1.0
    nt.links.new(color_out, sock["A_Color"])
    nt.links.new(ao.outputs["Color"], sock["B_Color"])
    return next(s for s in mix.outputs if s.identifier == "Result_Color")


def mask_view_image(mat):
    ts = mat.paintstack
    name = f"{pixels.PREFIX}{mat.name}_maskview"
    img = bpy.data.images.get(name)
    if img is None:
        img = pixels.new_image(f"{mat.name}_maskview", ts.res, (1.0, 1.0, 1.0))
    elif tuple(img.size) != (ts.res, ts.res):
        img.scale(ts.res, ts.res)
    return img


def _image_node(nt, label, image, location):
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.name = tex.label = NODE_PREFIX + label
    tex.image = image
    tex.location = location
    tex.hide = True
    return tex


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
    enabled = {c.id for c in enabled_channels(ts)}

    # Normal: the Normal channel (which already contains the baked normal), or the
    # baked normal on its own.
    normal_out = None
    normal_img = ts.output("NORMAL") if "NORMAL" in enabled else None
    if normal_img is None and ts.use_baked_normal:
        normal_img = ts.baked_map("NORMAL")
    if normal_img is not None:
        tex = _image_node(nt, "Normal" if "NORMAL" in enabled else "Baked Normal", normal_img,
                          (x - 300, y - 400))
        nmap = nt.nodes.new("ShaderNodeNormalMap")
        nmap.name = NODE_PREFIX + "Normal Map"
        nmap.location = (x, y - 400)
        nt.links.new(tex.outputs["Color"], nmap.inputs["Color"])
        normal_out = nmap.outputs["Normal"]
        _unlink(nt, bsdf.inputs["Normal"])
        nt.links.new(normal_out, bsdf.inputs["Normal"])

    texs = {}
    for ch in CHANNELS:
        img = ts.output(ch.id)
        if ch.id not in enabled or img is None or ch.id == "NORMAL":
            continue
        tex = texs[ch.id] = _image_node(nt, ch.label, img, (x, y))
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
        color_out = tex.outputs["Color"]
        if ch.id == "BASE_COLOR" and ts.preview_ao and ts.baked_map("AO") is not None:
            color_out = _multiply_ao(nt, color_out, ts.baked_map("AO"), (x + 300, y))
        nt.links.new(color_out, sock)
        if ch.id == "EMISSION" and "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = 1.0

    _build_view(mat, nt, texs, (bsdf.location.x, bsdf.location.y - 500))


def _build_view(mat, nt, texs, location):
    """Isolation view: route one channel (or the active layer's mask) straight to
    an emission output, so it shows unlit in the viewport."""
    ts = mat.paintstack
    mode = ts.view_mode
    user_outputs = [n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"
                    and not n.name.startswith(NODE_PREFIX)]
    if mode == "MATERIAL":
        if user_outputs and not any(n.is_active_output for n in user_outputs):
            user_outputs[0].is_active_output = True
        return

    if mode == "MASK":
        src = _image_node(nt, "Mask View", mask_view_image(mat), location).outputs["Color"]
    elif mode in texs:
        src = texs[mode].outputs["Color"]
    elif mode == "NORMAL" and ts.output("NORMAL") is not None and ts.use_normal:
        src = _image_node(nt, "Normal View", ts.output("NORMAL"), location).outputs["Color"]
    else:
        src = None

    emit = nt.nodes.new("ShaderNodeEmission")
    emit.name = NODE_PREFIX + "View"
    emit.location = (location[0] + 300, location[1])
    if src is not None:
        nt.links.new(src, emit.inputs["Color"])
    else:
        emit.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.name = NODE_PREFIX + "View Output"
    out.location = (location[0] + 500, location[1])
    nt.links.new(emit.outputs[0], out.inputs["Surface"])
    out.is_active_output = True
