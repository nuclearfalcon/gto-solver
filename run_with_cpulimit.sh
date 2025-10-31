#!/bin/bash
# CPU Limiting Wrapper for DCFR Research Scripts
#
# This script runs Python scripts with CPU usage limited (default: 80%) to prevent
# overheating and thermal throttling. Includes robust cleanup to prevent lingering
# processes after Ctrl+C interruption.
#
# Features:
#   - Configurable CPU limit (1-100%, default: 80%)
#   - Automatic CPU core detection
#   - Recursive process tree termination
#   - Three-layer cleanup (graceful → forced → pattern-based)
#   - Proper signal handling for Ctrl+C, SIGTERM, SIGINT
#
# Usage:
#   bash run_with_cpulimit.sh [OPTIONS] <script.py> [script args...]
#
# Options:
#   --cpu-limit N, -c N    Limit CPU to N% of total capacity (1-100, default: 80)
#
# Examples:
#   # Default 80% CPU limit
#   bash run_with_cpulimit.sh compare_dcfr_research_3p_parallel.py --iterations 100000
#
#   # Custom 70% CPU limit (long form)
#   bash run_with_cpulimit.sh --cpu-limit 70 compare_dcfr_research_3p_parallel.py --iterations 100000
#
#   # Conservative 50% CPU limit (short form)
#   bash run_with_cpulimit.sh -c 50 compare_dcfr_research_3p.py --iterations 1000000
#
# Requirements:
#   - cpulimit package: sudo apt-get install cpulimit
#   - OpenSpiel virtual environment: source ~/open_spiel/venv/bin/activate
#
# Cleanup:
#   Press Ctrl+C to interrupt. The script will automatically:
#   1. Send SIGTERM to all processes (graceful)
#   2. Wait 1 second for graceful shutdown
#   3. Send SIGKILL to any survivors (forced)
#   4. Pattern-based cleanup as last resort

set -e

# Default configuration
CPU_LIMIT_PERCENT=80  # Default: Limit to 80% of total CPU

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments for --cpu-limit flag
PYTHON_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu-limit|-c)
            if [ -z "$2" ] || [[ "$2" =~ ^- ]]; then
                echo -e "${RED}ERROR: --cpu-limit requires a numeric argument${NC}"
                echo ""
                echo "Usage: bash run_with_cpulimit.sh --cpu-limit N <script.py> [args...]"
                echo "Example: bash run_with_cpulimit.sh --cpu-limit 70 script.py --iterations 100000"
                exit 1
            fi
            CPU_LIMIT_PERCENT="$2"
            # Validate it's a number between 1-100
            if ! [[ "$CPU_LIMIT_PERCENT" =~ ^[0-9]+$ ]] || [ "$CPU_LIMIT_PERCENT" -lt 1 ] || [ "$CPU_LIMIT_PERCENT" -gt 100 ]; then
                echo -e "${RED}ERROR: CPU limit must be a number between 1 and 100${NC}"
                echo "Got: $CPU_LIMIT_PERCENT"
                exit 1
            fi
            shift 2
            ;;
        *)
            # All remaining arguments go to Python script
            PYTHON_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore arguments for Python script
set -- "${PYTHON_ARGS[@]}"

# Check if cpulimit is installed
if ! command -v cpulimit &> /dev/null; then
    echo -e "${RED}ERROR: cpulimit is not installed${NC}"
    echo ""
    echo "Please install cpulimit:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install cpulimit"
    echo ""
    exit 1
fi

# Check if OpenSpiel venv is activated
if [ -z "$VIRTUAL_ENV" ] || [[ "$VIRTUAL_ENV" != *"open_spiel"* ]]; then
    echo -e "${YELLOW}WARNING: OpenSpiel virtual environment may not be activated${NC}"
    echo "Expected virtual environment at: ~/open_spiel/venv"
    echo ""
    echo "To activate:"
    echo "  source ~/open_spiel/venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if at least one argument provided (the script to run)
if [ $# -lt 1 ]; then
    echo -e "${RED}ERROR: No script specified${NC}"
    echo ""
    echo "Usage:"
    echo "  bash run_with_cpulimit.sh [--cpu-limit N] <script.py> [args...]"
    echo "  bash run_with_cpulimit.sh [-c N] <script.py> [args...]"
    echo ""
    echo "Options:"
    echo "  --cpu-limit N, -c N    Limit CPU to N% of total (1-100, default: 80)"
    echo ""
    echo "Examples:"
    echo "  bash run_with_cpulimit.sh compare_dcfr_research_3p_parallel.py --iterations 100000"
    echo "  bash run_with_cpulimit.sh --cpu-limit 70 compare_dcfr_research_3p_parallel.py --iterations 100000"
    echo "  bash run_with_cpulimit.sh -c 50 compare_dcfr_research_3p.py --iterations 1000000"
    echo ""
    exit 1
fi

# Get number of CPU cores
NUM_CORES=$(nproc)

# Calculate CPU limit (percent * num_cores)
# cpulimit uses percentage PER CORE, so for 8 cores at 80% total = 640%
CPULIMIT_VALUE=$((CPU_LIMIT_PERCENT * NUM_CORES))

echo "========================================"
echo "CPU Limiting Wrapper"
echo "========================================"
echo "CPU cores:        $NUM_CORES"
echo "Total limit:      ${CPU_LIMIT_PERCENT}% of all cores"
echo "cpulimit value:   ${CPULIMIT_VALUE}%"
echo "Script:           $1"
echo "Arguments:        ${@:2}"
echo "========================================"
echo ""

# Store PID for cleanup
CPULIMIT_PID=""
CLEANUP_DONE=false

# Function to recursively kill process tree
kill_process_tree() {
    local pid=$1
    local sig=${2:-TERM}

    # Get all child PIDs
    local children=$(pgrep -P "$pid" 2>/dev/null)

    # Recursively kill children first
    for child in $children; do
        kill_process_tree "$child" "$sig"
    done

    # Kill the parent process
    if kill -0 "$pid" 2>/dev/null; then
        kill -$sig "$pid" 2>/dev/null || true
    fi
}

# Function to cleanup on exit
cleanup() {
    # Prevent multiple cleanup calls
    if [ "$CLEANUP_DONE" = true ]; then
        return
    fi
    CLEANUP_DONE=true

    # Disable exit-on-error for cleanup
    set +e

    echo ""
    echo -e "${YELLOW}Cleaning up all processes...${NC}"

    # Kill cpulimit and its entire process tree
    if [ -n "$CPULIMIT_PID" ] && kill -0 "$CPULIMIT_PID" 2>/dev/null; then
        echo "Stopping cpulimit and all child processes (PID: $CPULIMIT_PID)..."

        # First try graceful termination
        kill_process_tree "$CPULIMIT_PID" TERM
        sleep 1

        # Force kill any remaining processes
        kill_process_tree "$CPULIMIT_PID" KILL
    fi

    # Kill any remaining processes spawned by this script
    pkill -P $$ 2>/dev/null || true

    # Kill any python3 processes that match our script name (last resort)
    pkill -f "compare_dcfr_research_3p" 2>/dev/null || true

    echo -e "${GREEN}Cleanup complete${NC}"
}

# Register cleanup function
trap cleanup EXIT INT TERM SIGINT SIGTERM

# Run the Python script with cpulimit
echo -e "${GREEN}Starting with CPU limit of ${CPU_LIMIT_PERCENT}%...${NC}"
echo ""

# Execute with cpulimit in background to capture PID
# -l: limit percentage (per-core, so 80% of 8 cores = 640%)
# -f: monitor forks (follow child processes)
# --: separator before command
cpulimit -l ${CPULIMIT_VALUE} -f -- python3 "$@" &
CPULIMIT_PID=$!

# Wait for cpulimit to finish
wait $CPULIMIT_PID
EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Script completed successfully${NC}"
else
    echo -e "${RED}Script exited with code: $EXIT_CODE${NC}"
fi
echo "========================================"

exit $EXIT_CODE
