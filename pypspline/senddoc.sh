#!/usr/bin/env sh

cd html
tar cvfz pypspline_html.tgz *.htm *.html
scp pypspline_html.tgz pletzer@pypspline.sf.net:/home/groups/p/py/pypspline/htdocs
ssh pletzer@pypspline.sf.net "cd /home/groups/p/py/pypspline/htdocs; tar xvfz pypspline_html.tgz; cp index.htm index.html"
cd ..