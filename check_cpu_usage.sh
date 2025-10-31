#!/bin/bash
# Check actual CPU usage to verify cpulimit is working

echo "Monitoring CPU usage (press Ctrl+C to stop)..."
echo "This will show total CPU % every 2 seconds"
echo ""

while true; do
    # Get total CPU usage across all cores
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    
    # Count Python processes
    python_procs=$(ps aux | grep "compare_dcfr_research_3p_parallel.py" | grep -v grep | wc -l)
    
    echo "$(date +%T) | Total CPU: ${cpu_usage}% | Python processes: ${python_procs}"
    sleep 2
done
