#! /bin/bash

for i in `seq 1 3`; do
	echo ${i}
	docker cp robot-${i}:/pyfiles/logging-${i}.csv {PATH}
done
