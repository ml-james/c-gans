#!/bin/bash

pip install black
black --check .
if [ $? -ne 0 ];
then
        apt-get install git -y
        black .
        git add -A
        git commit -m "Black Changes"
        git push
fi
python3 -m unittest tests/*test*
