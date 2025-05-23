import mujoco
import mujoco.viewer
import mediapy as media
import numpy as np
import os
import re
import shutil
from pathlib import Path

def copy_mesh_files(urdf_path, mesh_paths):
    """Copy mesh files to the URDF directory"""
    urdf_dir = Path(urdf_path).parent
    copied_files = {}
    
    for package_path, absolute_path in mesh_paths.items():
        filename = Path(absolute_path).name
        target_path = urdf_dir / filename
        shutil.copy2(absolute_path, target_path)
        copied_files[package_path] = filename
    
    return copied_files

def resolve_package_path(package_path):
    """Convert ROS package:// paths to absolute paths"""
    match = re.match(r'package://([^/]+)/(.*)', package_path)
    if not match:
        return package_path
    
    package_name, rel_path = match.groups()
    workspace_path = Path("/workspace/ros2_ws")
    package_path = workspace_path / "src" / package_name
    
    if not package_path.exists():
        raise FileNotFoundError(f"Package directory not found: {package_path}")
    
    if "meshes/collision" in rel_path:
        collision_path = package_path / rel_path
        if collision_path.exists():
            return str(collision_path)
        visual_path = package_path / rel_path.replace("meshes/collision", "meshes/visual")
        if visual_path.exists():
            print(f"Warning: Using visual mesh for collision: {visual_path}")
            return str(visual_path)
    elif "meshes/visual" in rel_path:
        visual_path = package_path / rel_path
        if visual_path.exists():
            return str(visual_path)
    else:
        absolute_path = package_path / rel_path
        if absolute_path.exists():
            return str(absolute_path)
    
    raise FileNotFoundError(f"File not found: {package_path / rel_path}")

def convert_urdf_paths(urdf_path):
    """Convert package:// paths in URDF to relative paths"""
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()
    
    package_paths = re.findall(r'package://[^"]+', urdf_content)
    
    mesh_paths = {}
    for package_path in package_paths:
        try:
            absolute_path = resolve_package_path(package_path)
            mesh_paths[package_path] = absolute_path
        except FileNotFoundError as e:
            print(f"Warning: {str(e)}")
            if "collision" in package_path:
                urdf_content = re.sub(r'<collision>.*?</collision>', '', urdf_content, flags=re.DOTALL)
            else:
                urdf_content = re.sub(r'<visual>.*?</visual>', '', urdf_content, flags=re.DOTALL)
    
    copied_files = copy_mesh_files(urdf_path, mesh_paths)
    
    for package_path, filename in copied_files.items():
        urdf_content = urdf_content.replace(package_path, filename)
    
    temp_urdf_path = urdf_path + ".temp"
    with open(temp_urdf_path, 'w') as f:
        f.write(urdf_content)
    
    return temp_urdf_path

def main():
    try:
        urdf_path = "/workspace/ros2_ws/src/airhockey2025/ka/assets/main.xml"
        #temp_urdf_path = convert_urdf_paths(urdf_path)

        model = mujoco.MjModel.from_xml_path(urdf_path)
        data = mujoco.MjData(model)

        # Launch GUI viewer with a persistent loop
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("MuJoCo viewer launched. Close the window or press Ctrl+C to exit.")
            #print(model.nu)
            # qpos をランダムに
            
            Kp = 10.0
            Kd = 1.0
            i = 0
            try:
                while viewer.is_running():
                    if i== 0:
                        target_qpos = np.random.uniform(-np.pi, np.pi, model.nu)
                    q = data.qpos[:model.nu]
                    qd = data.qvel[:model.nu]
                    data.ctrl[:] = Kp * (target_qpos - q) - Kd * qd

                    mujoco.mj_step(model, data)
                    viewer.sync()
                    if i==9:
                        i=0
                    else:
                        i+=1
            except KeyboardInterrupt:
                print("Viewer closed by user.")

        # Cleanup after viewing
        # os.remove(temp_urdf_path)
        # urdf_dir = Path(urdf_path).parent
        # for file in urdf_dir.glob("*.stl"):
        #     os.remove(file)

    except Exception as e:
        print(f"Error: {str(e)}")
        # if temp_urdf_path:
        #     print(f"Temporary URDF file preserved at: {temp_urdf_path}")
        # raise


if __name__ == "__main__":
    main()

