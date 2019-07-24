#!/bin/sh
#
#echo "---------- Moving to repo ---------------------"
#cd automatic-film-detection

echo "---------- Capturing date and time ---------------------"
datetime=$(date '+%d/%m/%Y %H:%M:%S');

echo "-------------- Git pulling -----------------------------"
git pull origin master

echo "--------------- Adding all -----------------------------"
git add --all

echo "-------------- Making commit ---------------------------"
git commit -m "Updated: $datetime"

echo "------------ Pushing to server -------------------------"
git push -u origin master

echo "------------ Done --------------------------------------"
