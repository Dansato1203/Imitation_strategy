#! /bin/bash

for i in `seq 1 3`; do
	echo ${i}
	docker cp robot-${i}:/pyfiles/logging-${i}.csv /DATA/Lab./B4/221022/2
done
