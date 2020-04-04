# SACSANN
Sequence-based predictor of chromosomal compartments


## Installation

To install SACSANN, follow these steps:

*Clone this repo
```bash
git clone https://github.com/BlanchetteLab/SACSANN
```

* Go to the right folder `cd SACSANN/`

* SACCSANN supports Python 3. The recommended way to use it is to create a virtual
 environment :
```bash
pyenv virtualenv 3.7.3 sacsann
pyenv local sacsann
```
* then run `make install` to install the requirements

## Run the demo

You can run a demo of SACSANN on two mouse chromosomes by running `make run-demo`
For more information on the available options, run :
`python sacssann.py --help`

## Input

## Output

## License
LAMPS is free software: you can redistribute it and/or modify it under the terms of the 
GNU Lesser General Public License as published by the Free Software Foundation.

LAMPS is distributed in the hopes that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU Lesser General Public License for more details.
