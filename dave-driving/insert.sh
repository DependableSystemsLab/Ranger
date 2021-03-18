#!/bin/bash
config_file=$1 # yaml config file
file_to_insert=$2 # TF program file to insert Ranger
line=$3 # line of code to insert Ranger. NOTE: it must come after an non-empty line


# parse the config file to generate the code snippet to be inserted
python parse_config.py $config_file



code_file='ranger-code-to-insert.py' # code snippet to be inserted

#file_to_insert='huawei.py' # original ML program
#line=66  # line to insert Ranger. NOTE: This line needs to contain the code string for identifying the location for insertion
 
pointer=$(sed "${line}!d" "$file_to_insert")

sed -i.bak "/${pointer}/r ${code_file}" "${file_to_insert}"  
rm $code_file