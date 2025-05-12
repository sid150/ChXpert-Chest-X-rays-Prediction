import os
import chi

# Initialize the environment
context = chi.context
context.version = "1.0" 
context.choose_project()  # Select the correct project
context.choose_site(default="KVM@TACC")

# Get the username and prepare the environment
server_name = f"node-persist-project29"

# Function to list current volumes
def list_volumes():
    cinder_client = chi.clients.cinder()
    volumes = cinder_client.volumes.list()
    return volumes

# Function to create a new volume
def create_volume(name, size_gb):
    cinder_client = chi.clients.cinder()
    volume = cinder_client.volumes.create(name=name, size=size_gb)
    return volume

# Function to attach a volume to a server
def attach_volume_to_server(volume, server_name):
    server_id = chi.server.get_server(server_name).id
    volume_manager = chi.nova().volumes
    volume_manager.create_server_volume(server_id=server_id, volume_id=volume.id)
    return f"Volume {volume.name} attached to {server_name}"

# Function to detach a volume from a server
def detach_volume_from_server(volume, server_name):
    server_id = chi.server.get_server(server_name).id
    volume_manager = chi.nova().volumes
    volume_manager.delete_server_volume(server_id=server_id, volume_id=volume.id)
    return f"Volume {volume.name} detached from {server_name}"

# Function to delete a volume
def delete_volume(volume):
    cinder_client = chi.clients.cinder()
    cinder_client.volumes.delete(volume=volume)
    return f"Volume {volume.name} deleted"

# Example to create a volume, attach it, and delete it
if __name__ == "__main__":
    volume_name = f"block-persist-python-{username}"
    volume_size = 40  # in GiB

    # Create volume
    print("Creating volume...")
    volume = create_volume(volume_name, volume_size)
    print(f"Volume {volume.name} created with size {volume.size} GiB.")

    # Attach volume to the server
    print(f"Attaching volume {volume.name} to server {server_name}...")
    attachment_result = attach_volume_to_server(volume, server_name)
    print(attachment_result)

    # Optionally, detach and delete volume after usage
    print(f"Detaching volume {volume.name} from server {server_name}...")
    detach_result = detach_volume_from_server(volume, server_name)
    print(detach_result)

    print(f"Deleting volume {volume.name}...")
    delete_result = delete_volume(volume)
    print(delete_result)
