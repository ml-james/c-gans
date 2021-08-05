#!/bin/bash

pip install black
black --check .
if [ $? -ne 0 ];
then
        apt-get install git -y
        black .
        git add -A
        git commit -m "Black Changes"
        git config --global user.name 'Matthew James'
        git config --global user.email 'matt.james96@hotmail.co.uk'
        git remote set-url origin https://x-access-token:${{ secrets.GITHUB_TOKEN }}@github.com/${{ github.repository }}
        git push
fi
python3 -m unittest tests/*test*