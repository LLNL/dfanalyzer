#!/bin/sh

# check if all paths exist
raw_data_dir="$1"
raw_data_name="$2"
extracted_dir="$3"

# create raw file path
raw_data_file="$raw_data_dir/$raw_data_name.tar.gz"

# check if compressed file exists
if [ -f "$raw_data_file" ]; then
    if [ ! -d "$extracted_dir/$raw_data_name" ]; then
        mkdir -p "$extracted_dir/$raw_data_name"
        tar -xzf "$raw_data_file" -C "$extracted_dir/$raw_data_name"
    fi
fi
