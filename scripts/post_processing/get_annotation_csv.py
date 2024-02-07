import os
import pandas as pd

def find_mp4_folders(root_directory):
    mp4_folders = []

    for dirpath, dirnames, filenames in os.walk(root_directory):
        if 'MP4' in dirnames:
            mp4_folder_path = os.path.join(dirpath, 'MP4')
            mp4_folders.append(mp4_folder_path)

    return mp4_folders

directory_to_search = '/mnt/fsx/ashwinbalakrishna/datasets/TRI_goal_cond_narrow_eval/'
mp4_folders_list = list(reversed(sorted(find_mp4_folders(directory_to_search))))
mp4_folders_list = [path.split("TRI_goal_cond_narrow_eval/")[-1] for path in mp4_folders_list]
data = {"Video Path": mp4_folders_list, "Annotation": ['' for _ in range(len(mp4_folders_list))]}
df = pd.DataFrame(data)

csv_file_path = '/mnt/fsx/ashwinbalakrishna/datasets/TRI_goal_cond_narrow_eval/lang_annotations.csv'
df.to_csv(csv_file_path, index=False)
