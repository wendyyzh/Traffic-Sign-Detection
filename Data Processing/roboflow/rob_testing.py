from roboflow import Roboflow

# Initialize Roboflow client
rf = Roboflow(api_key="z0cr2Pq9nzZdnfIVptSz")

# Fetch the workspace
workspace = rf.workspace()

# List all projects in your workspace
print("Available projects in the workspace:")
projects = workspace.projects()

for project in projects:
    print(f"Project name: {project.name}, Project ID: {project.id}")