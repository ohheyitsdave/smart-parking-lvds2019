#!/bin/bash

set -e              # to make script exit as soon as any command fails

pushd `dirname $0` > /dev/null

echo "Build server"
virtualenv -p python3.7 ./runtime
source ./runtime/bin/activate
pip install -U -r requirements.txt

for i in `find src -iname "setup.py" -maxdepth 2`; do pip install -e `dirname ${i}`; done

deactivate

popd > /dev/null

