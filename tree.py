import os

def print_tree(root, file, prefix=""):
    file.write(prefix + os.path.basename(root) + "\n")
    prefix += "│   "
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            print_tree(path, file, prefix)
        else:
            file.write(prefix + name + "\n")

with open("directory_tree.txt", "w", encoding="utf-8") as f:
    print_tree(".", f)
