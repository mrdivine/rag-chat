#!/bin/bash

# Check if code_base.md exists in the current directory and delete it
if [[ -f "code_base.md" ]]; then
    echo "Removing existing code_base.md..."
    rm "code_base.md"
fi

# Function to print file content in markdown format
print_file_content() {
    local file=$1
    # Only process text files smaller than 1MB
    if [[ $(file --mime-type -b "$file") == text/* ]] && [[ $(stat -f%z "$file") -lt 1048576 ]]; then
        {
            printf "\`%s\`\n" "$file"
            printf '```\n'
            cat "$file"
            printf '```\n\n'
        }
    else
        {
            printf "\`%s\`\n" "$file"
            printf '```\n[Skipped: Non-text or too large]\n```\n\n'
        }
    fi
}

# Main function to recursively print files, excluding specified patterns
print_code_base() {
    local directory=${1:-.}  # Default to current directory if no argument is given

    # Patterns to exclude (add more patterns to this array as needed)
    local exclude_patterns=("*/mini_corpus/*" "*/.*" "*/__pycache__/*" "*/README.md" "./code_base.md")

    # Construct the find command with all exclude patterns and avoid symlinks
    local find_command="find \"$directory\" -maxdepth 10 -type f"
    for pattern in "${exclude_patterns[@]}"; do
        find_command+=" ! -path \"$pattern\""
    done

    # Output file
    local output_file="code_base.md"

    # Execute the find command and loop through files
    eval $find_command | while IFS= read -r file; do
        # Check if the file actually exists
        if [[ -f "$file" ]]; then
            # Append content directly to the output file
            print_file_content "$file" >> "$output_file"
        fi
    done

    echo "Code base saved to $output_file."
}

# Execute the function with the first argument or without arguments
print_code_base "$1"
