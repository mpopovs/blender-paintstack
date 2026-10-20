"""Layer hierarchy. Layers stay in one flat, top-first collection; a folder is
immediately followed by its descendants (each layer names its folder in
parent_uid), so every folder and its contents form one contiguous block."""


def by_uid(ts):
    return {l.uid: l for l in ts.layers}


def ancestors(ts, layer, lookup=None):
    lookup = lookup or by_uid(ts)
    out = []
    p = lookup.get(layer.parent_uid)
    while p is not None:
        out.append(p)
        p = lookup.get(p.parent_uid)
    return out


def depth(ts, layer, lookup=None):
    return len(ancestors(ts, layer, lookup))


def is_inside(ts, layer, folder_uid, lookup=None):
    return any(a.uid == folder_uid for a in ancestors(ts, layer, lookup))


def block_end(ts, index):
    """Exclusive end of the block starting at index (the layer plus its descendants)."""
    uid = ts.layers[index].uid
    lookup = by_uid(ts)
    j = index + 1
    while j < len(ts.layers) and is_inside(ts, ts.layers[j], uid, lookup):
        j += 1
    return j


def tree(ts, start=0, end=None, parent=""):
    """Nested [(layer, children)] for the layers in [start, end), top-first."""
    end = len(ts.layers) if end is None else end
    out = []
    i = start
    while i < end:
        layer = ts.layers[i]
        j = block_end(ts, i)
        if layer.parent_uid == parent:
            out.append((layer, tree(ts, i + 1, j, layer.uid) if layer.type == "FOLDER" else []))
        i = j
    return out


def hidden_by_collapse(ts, layer, lookup=None):
    return any(not a.expanded for a in ancestors(ts, layer, lookup))


def reorder(ts, uids):
    """Rearrange the collection into the given uid order, keeping the active layer."""
    active = ts.active_layer.uid if ts.active_layer else None
    for pos, uid in enumerate(uids):
        cur = next(i for i, l in enumerate(ts.layers) if l.uid == uid)
        if cur != pos:
            ts.layers.move(cur, pos)
    if active:
        ts.active_layer_index = next(i for i, l in enumerate(ts.layers) if l.uid == active)


def move(ts, index, direction):
    """Move the layer block at index one step among its siblings. Stepping past an
    expanded sibling folder enters it; stepping past the edge of a folder leaves it.
    Returns False when the layer cannot move further."""
    layers = ts.layers
    n = len(layers)
    layer = layers[index]
    end = block_end(ts, index)
    uids = [l.uid for l in layers]
    block = uids[index:end]
    rest = uids[:index] + uids[end:]
    lookup = by_uid(ts)

    if direction == "UP":
        # Nearest layer above that is a sibling (start of its block), or the parent folder.
        k = index - 1
        while k >= 0 and layers[k].parent_uid != layer.parent_uid and layers[k].uid != layer.parent_uid:
            k -= 1
        if k < 0:
            return False
        above = layers[k]
        new_parent = layer.parent_uid
        if above.uid == layer.parent_uid:
            # First child of a folder: leave it, landing just above the folder.
            new_parent = above.parent_uid
            insert = rest.index(above.uid)
        elif above.type == "FOLDER" and above.expanded:
            # Enter the expanded folder above as its last child.
            new_parent = above.uid
            insert = rest.index(uids[block_end(ts, k) - 1]) + 1
        else:
            insert = rest.index(above.uid)
    else:
        if end >= n or (layers[end].parent_uid != layer.parent_uid):
            parent = lookup.get(layer.parent_uid)
            if parent is None:
                return False
            # Last child of a folder: leave it, landing just below the folder's block.
            new_parent = parent.parent_uid
            p_index = next(i for i, l in enumerate(layers) if l.uid == parent.uid)
            p_end = block_end(ts, p_index)
            # Our block is the tail of the folder's block, so land after it.
            insert = len(rest[:p_end - len(block)])
        else:
            below = layers[end]
            new_parent = layer.parent_uid
            if below.type == "FOLDER" and below.expanded:
                new_parent = below.uid
                insert = rest.index(below.uid) + 1
            else:
                insert = rest.index(uids[block_end(ts, end) - 1]) + 1
    layer.parent_uid = new_parent
    reorder(ts, rest[:insert] + block + rest[insert:])
    return True
