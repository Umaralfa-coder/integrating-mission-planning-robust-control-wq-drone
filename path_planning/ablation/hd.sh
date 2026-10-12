#!/bin/sh
cd /home/claude/covplan_run8/wsweep
for a in aoi8 aoi4; do for S in 4.0 3.2; do python3 job3.py $a 90 1 $S > hd_${a}_$S.log 2>&1 & done; done
wait
