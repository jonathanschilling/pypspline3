#!/usr/bin/env sh

cd pypspline
for f in `ls *.py`; do
    f2=`echo $f | awk -F'.' '{print $1}'`
    pydoc -w $f2
done
rm __init__.html
cp *.html ../html
cd ..
cd html
tar cvfz pypspline_html.tgz *.htm *.html
scp pypspline_html.tgz pletzer@pypspline.sf.net:/home/groups/p/py/pypspline/htdocs
ssh pletzer@pypspline.sf.net "cd /home/groups/p/py/pypspline/htdocs; tar xvfz pypspline_html.tgz; cp index.htm index.html"
cd ..

echo "to upload a file release:"
echo "ftp upload.sourceforge.net (anonymous)"
echo "cd /incoming"
echo "bin"
echo "hash"
echo "mput dist/pypspline*.tar.gz"
