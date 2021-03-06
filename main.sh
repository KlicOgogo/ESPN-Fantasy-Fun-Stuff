#!/usr/bin/env bash
set -ex
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

git reset --hard
git clean -fd
git checkout master
git pull origin master
./src/main.py
git add *.html
git commit -m "New matchup stats"
git push origin master
