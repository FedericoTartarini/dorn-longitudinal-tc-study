import os

main_dir = os.path.dirname(os.getcwd())
git_path = os.path.join(main_dir, ".git")

for root, dirs, files in os.walk(git_path):
    for file in files:
        if ".ini" in file:
            os.remove(os.path.join(root, file))
            # print(root, file)
