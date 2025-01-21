from clearml import Task, StorageManager

# Connect to the ClearML task using the task ID
task_id = "66508dcde37b4d8b8c6705a1284484a6"  # Replace with your task ID
task = Task.get_task(task_id=task_id)

# List the artifacts associated with the task
artifacts = task.artifacts
print("Available Artifacts:")
for artifact_name, artifact in artifacts.items():
    print(f" - {artifact_name}: {artifact.url}")

# Download the model artifact if it exists
if "model" in artifacts:  # Replace "model" with the artifact name if it's different
    model_url = artifacts["model"].url
    local_path = StorageManager.get_local_copy(model_url)
    print(f"Model saved to: {local_path}")
else:
    print("No model artifact found in this task.")
