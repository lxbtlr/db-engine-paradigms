#!/bin/bash
# Grant cap_sys_nice to all ELF executables in build subdirectories,
# allowing chrt -f 99 without root at runtime.
# Usage: sudo ./setcap.sh [build_dir]

BUILD_DIR="${1:-build}"

if [ "$(id -u)" -ne 0 ]; then
    echo "Must run as root (sudo ./setcap.sh)" >&2
    exit 1
fi

count=0
while IFS= read -r -d '' f; do
    file "$f" | grep -q 'ELF.*executable' && {
        setcap 'cap_sys_nice=eip' "$f"
        echo "  $f"
        ((count++))
    }
done < <(find "$BUILD_DIR" -maxdepth 3 -type f -executable -print0)

echo "Set cap_sys_nice on $count binaries."
