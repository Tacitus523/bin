#!/bin/bash

user=$USER
# argparser to pass -u user
while getopts ":u:" opt; do
  case ${opt} in
    u )
      user=$OPTARG
      ;;
    \? )
      echo "Usage: cmd [-u user]"
      exit 1
      ;;
  esac
done


RED='\E[47;0;32m'
NC='\e[0m'

BOLD='\033[1m'
PLAIN='\033[0m'

IFS=

qstat -q '*' -u "$user" -xml $*| qstat.awk | \
while read line
do
	#echo "$line"
	if echo $line | grep $USER > /dev/null
	then
		line="$RED$line$NC"
		line="$BOLD$line$PLAIN"
	fi
	echo -e "$line"
done