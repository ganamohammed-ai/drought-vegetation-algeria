import tifffile, numpy as np, zlib
def read_chirps(path):
    with tifffile.TiffFile(path) as t:
        p=t.pages[0]; W=p.imagewidth; H=p.imagelength; bps=4
        tw,th=(p.tilewidth,p.tilelength) if p.is_tiled else (W,p.rowsperstrip)
        offs,cnts=p.dataoffsets,p.databytecounts
        fh=t.filehandle
        ntx=-(-W//tw); nty=-(-H//th)
        out=np.full((nty*th,ntx*tw),np.nan,np.float32)
        for k,(o,c) in enumerate(zip(offs,cnts)):
            fh.seek(o); raw=fh.read(c); raw=zlib.decompress(raw)
            b=np.frombuffer(raw,dtype=np.uint8).reshape(th,tw*bps)
            b=np.cumsum(b,axis=1,dtype=np.uint8).reshape(th,bps,tw).transpose(0,2,1)
            v=np.ascontiguousarray(b).view('>f4').reshape(th,tw).astype(np.float32)
            ty,tx=divmod(k,ntx); out[ty*th:(ty+1)*th,tx*tw:(tx+1)*tw]=v
        a=out[:H,:W]
        sc=p.tags[33550].value; tp=p.tags[33922].value
        nd=p.tags[42113].value if 42113 in p.tags else None
    return a,(float(sc[0]),float(sc[1])),(float(tp[3]),float(tp[4])),nd
