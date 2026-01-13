import os
import trimesh

def convert_stl_to_obj():
    # 获取当前文件夹中的所有 STL 文件
    stl_files = [f for f in os.listdir() if f.endswith('.STL')]

    if not stl_files:
        print("No STL files found in the current folder.")
        return

    for stl_file in stl_files:
        # 读取 STL 文件
        mesh = trimesh.load_mesh(stl_file)

        # 构造输出 OBJ 文件名
        obj_file = os.path.splitext(stl_file)[0] + ".obj"

        # 保存为 OBJ 格式
        mesh.export(obj_file)
        print(f"Converted: {stl_file} -> {obj_file}")

if __name__ == "__main__":
    convert_stl_to_obj()