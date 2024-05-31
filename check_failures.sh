#Give folder-prefix as $1 and file-prefix as $2
set -o errexit   # (or set -e) cause batch script to exit immediately when a command fails.

for folder in $1*/; do if ! grep -q -m 1 "FINAL SINGLE" $folder$2.out 2> /dev/null; then echo "$folder failed"; fi; done
