import os

# Set W&B API key
os.environ['WANDB_API_KEY'] = 'da30da01fd3e0628233dc693966e900058ff208e'
# Verify it is set
print(f"W&B API Key in script: {os.environ.get('WANDB_API_KEY')}")