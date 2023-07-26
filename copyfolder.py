import os
import shutil

def copy_folder(source_folder, destination_folder):
    shutil.copytree(source_folder, destination_folder)

# 소스 폴더 경로
src = '/home/irteam/rkdtjdals97-dcloud-dir/co/amos22'

# 대상 폴더 경로
dst = 'amos22'

# os.makedirs(dest, exist_ok=True)

# 폴더 복사
copy_folder(src, dst)