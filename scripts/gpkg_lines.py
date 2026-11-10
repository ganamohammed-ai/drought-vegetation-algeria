import sqlite3, struct
def read_polys(path):
    con = sqlite3.connect(path)
    t = con.execute("select table_name from gpkg_contents").fetchone()[0]
    out = []
    for (g,) in con.execute(f"select geom from '{t}'"):
        flags = g[3]; env = (flags >> 1) & 7
        off = 8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[env]
        out.extend(_wkb(g, off)[0])
    return out
def _wkb(b, o):
    bo = '<' if b[o] == 1 else '>'; o += 1
    typ = struct.unpack(bo + 'I', b[o:o+4])[0] & 0xff; o += 4
    polys = []
    if typ == 3:
        o, rings = _rings(b, o, bo); polys.append(rings)
    elif typ == 6:
        n = struct.unpack(bo + 'I', b[o:o+4])[0]; o += 4
        for _ in range(n):
            p, o = _wkb(b, o); polys.extend(p)
    else:
        raise ValueError(typ)
    return polys, o
def _rings(b, o, bo):
    nr = struct.unpack(bo + 'I', b[o:o+4])[0]; o += 4
    rings = []
    for _ in range(nr):
        n = struct.unpack(bo + 'I', b[o:o+4])[0]; o += 4
        import array
        a = struct.unpack(bo + 'd' * (2 * n), b[o:o+16*n]); o += 16 * n
        rings.append((a[0::2], a[1::2]))
    return o, rings
