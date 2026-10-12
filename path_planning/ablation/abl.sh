#!/bin/sh
cd "$(dirname "$0")"
python3 job2.py aoi8 90 1 "(0.0,0.0,1.0,0.0)" &
python3 job2.py aoi4 90 1 "(0.0,0.0,1.0,0.0)" &
wait
