import os

def print_tree(root, prefix=""):
    files = os.listdir(root)
    for index, name in enumerate(sorted(files)):
        path = os.path.join(root, name)
        connector = "└── " if index == len(files) - 1 else "├── "
        print(prefix + connector + name)
        if os.path.isdir(path):
            extension = "    " if index == len(files) - 1 else "│   "
            print_tree(path, prefix + extension)

print_tree('/home/tuandang/tuandang/quanganh/visualnav-transformer/train')
