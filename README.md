Typesetter
==========

Tool for converting the medial axis transform font curves for carving with a V-bit.

Typesetter uses freetype to render text in a selected font, and then scikit-image (a scientific imaging library) to generate the medial axis of the rendered bitmap.

At present, this is just a concept example, and is not really polished enough for use in real carving applications without a lot of tinkering and hand-holding.

Usage
-----
To run the example, simply `python main.py`

Screenshots
-----------
![Text](https://raw.github.com/ryansturmer/typesetter/master/images/text.png)
*Original Image*

![Pretty](https://raw.github.com/ryansturmer/typesetter/master/images/pretty.png)
*Contrast-stretched Medial Axis*

![3D View](https://raw.github.com/ryansturmer/typesetter/master/images/3d.png)
*3D Toolpath View*
