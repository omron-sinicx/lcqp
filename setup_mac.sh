#!/bin/zsh

git submodule update --init --recursive
cd LCQPow
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_DOCUMENTATION=OFF -DUNIT_TESTS=OFF 
make -j4
cd ../..
CURR_PATH=$(cd $(dirname $0);pwd)
echo "export PYTHONPATH=$CURR_PATH:$CURR_PATH/LCQPow/build/interfaces/python:$PYTHONPATH" >> ~/.zshrc
python3 -m pip install -r requirements.txt
source ~/.zshrc