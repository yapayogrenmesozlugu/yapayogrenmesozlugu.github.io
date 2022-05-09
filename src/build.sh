#!/bin/bash

rm -rf _build
d2lbook build html

python3 revise_htmls.py

cp edited_search.html _build/html/search.html
cp edited_searchtools.js _build/html/_static/searchtools.js
