#!/bin/zsh

# This just gets all of the samples from Google Storage.

mkdir -p samples

IFS=$'\n\n'
set -f

for i in $(cat < "sample_names.txt"); do
  curl -o "./samples/${${i}// /_}.npy" "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/${${i}// /%20}.npy"
done