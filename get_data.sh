#!/bin/bash

wget "https://zenodo.org/records/3360392/files/D184MB.zip?download=1" -O books.zip
unzip "books.zip"
mkdir -p books
mv D184MB/* books/
rm -r D184MB
rm books.zip