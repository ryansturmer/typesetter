from numpy import zeros, array, int8, logical_or, copyto, maximum, greater

image = array(
    [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
    dtype=int8
    )

cross = array([
    [0,1,0],
    [1,1,1],
    [0,1,0],
    ],
    dtype=int8)

def expand(image, N):
    'Returns the provided image, with its border expanded by N pixels on each side.'
    rows,cols = image.shape
    new_image = zeros((rows+(2*N),cols+(2*N)), dtype=int8)
    new_image[N:N+rows,N:N+cols] = image
    return new_image

def invert(image):
    return abs(1-image)

def dilate(img, keep_dims=False):
    print keep_dims
    img = expand(img, 1)
    rows, cols = img.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if(img[i,j] == 1):
                img[i+1,j] = 2 if img[i+1,j] == 0 else img[i+1,j]
                img[i-1,j] = 2 if img[i-1,j] == 0 else img[i-1,j]
                img[i,j+1] = 2 if img[i,j+1] == 0 else img[i,j+1]
                img[i,j-1] = 2 if img[i,j-1] == 0 else img[i,j-1]

    img = greater(img, 0).astype(int8)
    if keep_dims:
        return img[1:rows-1][1:cols-1]
    else:
        return img

def erode(img, keep_dims=False):
    return invert(dilate(invert(img), keep_dims=keep_dims))

def sesum(a, n):
    for i in range(n):
        a = dilate(a, keep_dims=True)
    return a


def open(image):
    return dilate(erode(image))

def openth(image):
    return image - open(image)

def union(a,b):
    return maximum(a,b)

def mmskelm(f,B=cross):
    y = f
    for i in range(max(f.shape)):
        print i
        print "Sesum"
        nb = sesum(cross, i)
        print "erode"
        f1 = erode(f, nb)
        print "openth"
        f2 = openth(f,B)
        print "union"
        y = union(y, f2)

'''
function y=mmskelm_equ(f, B)
  y = mmbinary(zeros(size(f)));
  for i=0:length(f)
    nb = mmsesum(B,i);
    f1 = mmero(f,nb);
    f2 = mmopenth(f1,B);
    y = mmunion(y,f2);
  end; 
'''

print image
print mmskelm(cross, 10)