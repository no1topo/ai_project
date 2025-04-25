import subprocess

# Number of times to run the script
num_processes = 8

# List to store subprocesses
processes = []

# Run the script 12 times
for _ in range(num_processes):
    process = subprocess.Popen(["python", "selfplay.py"])
    processes.append(process)

# Wait for all processes to complete
for process in processes:
    process.wait()

print("All instances of selfplay.py have completed.")