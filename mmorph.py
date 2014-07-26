from numpy import zeros, array, int8, logical_or, copyto, maximum

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
    rows,cols = image.shape
    new_image = zeros((rows+(2*N),cols+(2*N)), dtype=int8)
    new_image[N:N+rows,N:N+cols] = image
    return new_image

def invert(image):
    return abs(1-image)

def minkowski(a, b, sign=0):
    ssize = b.shape[0]/2
    output_image = expand(a, ssize)
    input_image = expand(a, ssize)
    if(sign):
        output_image = invert(output_image)
        input_image = invert(input_image)

    rows, cols = output_image.shape
    for i in range(ssize, rows-ssize):
        for j in range(ssize, cols-ssize):
            if(input_image[i,j]):
                output_image[i-ssize:i+ssize+1, j-ssize:j+ssize+1] |=  b

    if(sign):
        return invert(output_image)
    else:
        return output_image

def sesum(a, n):
    print a
    for i in range(n):
        print "  ", i
        a = minkowski(a,a)
    print a
    return a

def dilate(image, structuring_element):
    ssize = structuring_element.shape[0]/2
    tmp = minkowski(image, structuring_element, sign=0)
    rows, cols = tmp.shape
    return tmp[ssize:rows-ssize, ssize:cols-ssize]

def erode(image, structuring_element):
    ssize = structuring_element.shape[0]/2
    tmp = minkowski(image, structuring_element, sign=1)
    rows, cols = tmp.shape
    return tmp[ssize:rows-ssize, ssize:cols-ssize]

def open(image, structuring_element):
    return dilate(erode(image, structuring_element), structuring_element)

def openth(image, structuring_element):
    return image - open(image, structuring_element)

def union(a,b):
    return maximum(a,b)

def mmskelm(f,B=cross):
    y = f
    for i in range(max(f.shape)):
        print i
        print "Sesum"
        nb = sesum(B, i)
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
print mmskelm(image)