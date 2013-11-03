import skimage.morphology
from freetype import *
import numpy as np
from skimage import morphology, data, io, img_as_bool, img_as_float, filter
def POINT_SIZE(x):
    return x*64

def get_text_extents(text, filename, size=50):
    face = Face(filename)
    face.set_char_size(POINT_SIZE(size))
    flags = FT_LOAD_RENDER
    pen = FT_Vector(0,0)
    previous = None # 0
    xmin, xmax = 0, 0
    ymin, ymax = 0, 0
    for c in text:
        face.load_char(c, flags)
        kerning = face.get_kerning(previous, c)
        previous = c
        bitmap = face.glyph.bitmap
        pitch  = face.glyph.bitmap.pitch
        width  = face.glyph.bitmap.width
        rows   = face.glyph.bitmap.rows
        top    = face.glyph.bitmap_top
        left   = face.glyph.bitmap_left
        pen.x += kerning.x
        x0 = (pen.x >> 6) + left
        x1 = x0 + width
        y0 = (pen.y >> 6) - (rows - top)
        y1 = y0 + rows
        xmin, xmax = min(xmin, x0),  max(xmax, x1)
        ymin, ymax = min(ymin, y0), max(ymax, y1)
        pen.x += face.glyph.advance.x
        pen.y += face.glyph.advance.y
    return xmin, xmax, ymin, ymax

def get_text(text, filename, size):
    face = Face(filename)
    face.set_char_size(POINT_SIZE(size))
    pen = FT_Vector(0,0)
    flags = FT_LOAD_RENDER
    xmin,xmax,ymin,ymax = get_text_extents(text, filename, size)
    L = np.zeros((ymax-ymin, xmax-xmin),dtype=np.ubyte)
    previous = 0
    pen.x, pen.y = 0, 0
    for c in text:
        face.load_char(c, flags)
        kerning = face.get_kerning(previous, c)
        previous = c
        bitmap = face.glyph.bitmap
        pitch  = face.glyph.bitmap.pitch
        width  = face.glyph.bitmap.width
        rows   = face.glyph.bitmap.rows
        top    = face.glyph.bitmap_top
        left   = face.glyph.bitmap_left
        pen.x += kerning.x
        x = (pen.x >> 6) - xmin + left
        y = (pen.y >> 6) - ymin - (rows - top)
        data = []
        for j in range(rows):
            data.extend(bitmap.buffer[j*pitch:j*pitch+width])
        if len(data):
            Z = np.array(data,dtype=np.ubyte).reshape(rows, width)
            L[y:y+rows,x:x+width] |= Z[::-1,::1]
        pen.x += face.glyph.advance.x
        pen.y += face.glyph.advance.y
    return L

image = get_text("Test", 'sandbox/freetype-examples/Vera.ttf', 500)
thresh = filter.threshold_otsu(image)
print thresh
binary = image > thresh
print binary
skel, distance = morphology.medial_axis(binary, return_distance=True)
#skel = morphology.skeletonize(binary)
io.imsave('test.png', img_as_float(skel))
