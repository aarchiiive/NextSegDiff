import wandb
import pandas as pd
import time

def log_wandb(filename, dataset, project):
    # CSV 파일을 DataFrame으로 읽어옵니다.
    data = pd.read_csv(filename)

    # wandb 초기화
    wandb.init(name=dataset, project=project)

    # 각 행을 하나씩 wandb에 로그로 기록합니다.
    for index, row in data.iterrows():
        log_data = {}
        
        for k in row.to_dict().keys():
            log_data[k] = row[k]
            print(type(row[k]))

        # log_data를 wandb에 기록합니다.
        wandb.log(log_data, step=index)

    # wandb 실행 종료
    wandb.finish()

if __name__ == "__main__":
    # log_wandb('amos/outputs/progress.csv', 'AMOS', 'medsegdiff')
    log_wandb('amos22/outputs/progress.csv', 'AMOS_2022_3D', 'medsegdiff')
    # import os
    # print(len(os.listdir('amos/imagesTr')))
