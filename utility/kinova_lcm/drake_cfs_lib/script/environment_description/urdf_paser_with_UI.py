import xml.etree.ElementTree as ET
import ipdb
import tkinter as tk
from tkinter import messagebox

def update_model_position(model_name, new_xyz, new_rpy):
    urdf_file = 'environment.urdf'
    # Parse the URDF file
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    # Find the visual and collision elements for the given model
    visual_model = root.find(f".//visual[@name='{model_name}']/origin")
    collision_model = root.find(f".//collision[@name='{model_name}']/origin")

    if visual_model is not None:
        visual_model.set('xyz', new_xyz)  # Update visual position
        visual_model.set('rpy', new_rpy)  # Update visual position
    else:
        print(f"Error: Visual '{model_name}' not found.")

    if collision_model is not None:
        collision_model.set('xyz', new_xyz)  # Update collision position
        collision_model.set('rpy', new_rpy)  # Update visual position
    else:
        print(f"Error: Collision '{model_name}' not found.")

    # Save the updated URDF to a new file
    tree.write(urdf_file)
    print(f"Updated URDF saved to {urdf_file}")


# Function to parse, add sphere, and return updated URDF as a string
def add_sphere_to_urdf(name, new_xyz, radius=0.3):
    urdf_file = 'environment.urdf'
    # Parse the original URDF
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    # Find the "robot_base" link element to add the new visual and collision
    robot_base_link = root.find(".//link[@name='robot_base']")

    # Check if a visual or collision with the specified name already exists
    if robot_base_link.find(f"./visual[@name='{name}']") is not None or \
       robot_base_link.find(f"./collision[@name='{name}']") is not None:
        print(f"Element with name '{name}' already exists. No changes made.")
        return

    # Define the visual element for the sphere
    visual = ET.SubElement(robot_base_link, "visual", name=name)
    ET.SubElement(visual, "origin", rpy="0 0 0", xyz=new_xyz)
    geometry_visual = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry_visual, "sphere", radius=str(radius))
    material_visual = ET.SubElement(visual, "material", name="CFSObstacle_material")
    ET.SubElement(material_visual, "color", rgba="0.0 0.0 0.64 1.0")

    # Define the collision element for the sphere
    collision = ET.SubElement(robot_base_link, "collision", name=name)
    ET.SubElement(collision, "origin", rpy="0 0 0", xyz=new_xyz)
    geometry_collision = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry_collision, "sphere", radius=str(radius))
    dissipation = ET.SubElement(geometry_collision, "dissipation")
    dissipation.text = "1"

    # Save the updated URDF to a new file
    tree.write(urdf_file)
    print(f"Updated URDF saved to {urdf_file}")



# # Placeholder function for updating the model position
# def update_model_position(model_name, new_xyz):
#     # Replace this function with actual logic to update the model's position
#     print(f"Model: {model_name}, New Position: {new_xyz}")
#     # Feedback message
#     messagebox.showinfo("Update Successful", f"Model '{model_name}' position updated to {new_xyz}.")

# Function to handle the button click event
def on_update_click():
    model_name = model_name_entry.get()
    new_xyz = new_xyz_entry.get()
    new_rpy = new_rpy_entry.get()
    if model_name and new_xyz:
        print(f"Model: {model_name}, New Position: {new_xyz}, New Orientation: {new_rpy}")
        update_model_position(model_name, new_xyz, new_rpy)
    else:
        messagebox.showwarning("Input Error", "Please fill in both fields.")

# Create the main window
root = tk.Tk()
root.title("CFS Scene Configuration UI")

# Model name label and entry
tk.Label(root, text="Model Name:").grid(row=0, column=0, padx=10, pady=5)
model_name_entry = tk.Entry(root)
model_name_entry.grid(row=0, column=1, padx=10, pady=5)

# New XYZ label and entry
tk.Label(root, text="New XYZ Position:").grid(row=1, column=0, padx=10, pady=5)
new_xyz_entry = tk.Entry(root)
new_xyz_entry.grid(row=1, column=1, padx=10, pady=5)

# New XYZ label and entry
tk.Label(root, text="New RPY Orientation:").grid(row=2, column=0, padx=10, pady=5)
new_rpy_entry = tk.Entry(root)
new_rpy_entry.grid(row=2, column=1, padx=10, pady=5)

# Update button
update_button = tk.Button(root, text="Update", command=on_update_click)
update_button.grid(row=3, columnspan=2, pady=10)

# Run the application
root.mainloop()


# urdf_file = 'environment.urdf'  # Replace with your URDF file path
# # Example for change obstacle positions
# model_name = "ball1"  # Replace with the model you want to modify
# new_xyz = "4 1.5 2"  # New position
# update_model_position(model_name, new_xyz)


# mode = "1" # consider to change object position

# # Example for add a new obstacle
# # model_name = "ball3"  # Replace with the model you want to modify
# # new_xyz = "1 2 1"  # New position
# # mode = "2"    # consider to add new object 

# if mode == "1":
#     # Update the model position
#     update_model_position(model_name, new_xyz)
# elif mode == "2":
#     # Call the function and print the updated URDF
#     add_sphere_to_urdf(model_name, new_xyz)
