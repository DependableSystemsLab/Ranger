#!/bin/bash
config_file=$1 # yaml config file
file_to_insert=$2 # TF program file to insert Ranger
line=$3 # line of code to insert Ranger. NOTE: it must come after an non-empty line
outFile=$4 # yaml file to output the restriction values

# parse the config file to generate the code snippet to be inserted
python parse_profiling_config.py $config_file $outFile



profiling_file='profiling_filled_template.py' # code snippet to be inserted


pointer=$(sed "${line}!d" "$file_to_insert")

sed -i.bak "/${pointer}/r ${profiling_file}" "${file_to_insert}"  
rm $profiling_file


