#!/bin/bash

rm -rf _build
d2lbook build html

python3 revise_htmls.py
