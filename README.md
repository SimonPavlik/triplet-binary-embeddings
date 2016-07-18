# Paper: "Fast Training of Triplet-based Deep Binary Embedding Networks"

##People
Bohan Zhuang, Guosheng Lin, Chunhua Shen and Ian Reid.
Code author: Bohan Zhuang
This code is provided for non-profit research purpose only; and is released under the GNU license. 
For commercial applications, please contact Chunhua Shen http://www.cs.adelaide.edu.au/~chhshen/.

__This is the implementation of the following paper. If you use this code in your research, please cite our paper__

```

@inproceedings{zhuang2016fast,
  title={Fast Training of Triplet-based Deep Binary Embedding Networks},
  author={Zhuang, Bohan and Lin, Guosheng and Shen, Chunhua and Reid, Ian},
  journal={arXiv preprint arXiv:1603.02844},
  year={2016}
}

```

## Overview
./step1/ includes the code for the binary codes inference step. 
./lib/ includes the necessary codes for the network training. We inplement it using Theano. 
./preprocessing is the data preprocessing toolbox. 

## Training

The code is based on Ubuntu 14.04.
The main function is the ./step1/train.m file.
Modify the paths in the above main file and config.yaml.



## Copyright

Copyright (c) Bohan Zhuang. 2016.

** This code is for non-commercial purposes only. For commerical purposes,
please contact Chunhua Shen <chhshen@gmail.com> **

This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
