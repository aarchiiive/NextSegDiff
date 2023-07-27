import requests
from tqdm import tqdm

def get_data(url):
    filename = url.split("/")[-1]
    print("filename :", filename)
    response = requests.get(url, stream=True)

    # 데이터 저장
    file_size = int(response.headers.get('Content-Length', 0))
    block_size = 1024  # 1KB 단위로 다운로드 진행 상황 표시
    progress_bar = tqdm(total=file_size, unit='B', unit_scale=True)

    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()
    print('Download Complete!')
    
if __name__ == "__main__":
    # 2020
    # # training data
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_Dicom.zip")
    # # training ground truth
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth.csv")
    # # testing data
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Test_Dicom.zip")
    
    # # training data(JPEG)
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip")
    # # testing data(JPEG)
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Test_JPEG.zip")
    
    # 2018
    # # training data
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip")
    # # training ground truth
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip")
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task2_Training_GroundTruth_v3.zip")
    # # testing data
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Test_Input.zip")
    # # testing ground truth
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Test_GroundTruth.zip")
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task2_Test_GroundTruth.zip")
    
    # 2016
    # training data
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_Data.zip")
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part2_Training_Data.zip")
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part2B_Training_Data.zip")
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_Data.zip")
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3B_Training_Data.zip")
    # # training ground truth
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_GroundTruth.zip")
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part2_Training_GroundTruth.zip")
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part2B_Training_GroundTruth.zip")
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_GroundTruth.csv")
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3B_Training_GroundTruth.csv")
    # # testing data
    get_data("https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_Data.zip")
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part2_Test_Data.zip")
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part2B_Test_Data.zip")
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_Data.zip")
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3B_Test_Data.zip")
    # # testing ground truth
    get_data("https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_GroundTruth.zip")
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part2_Test_GroundTruth.zip")
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part2B_Test_GroundTruth.zip")
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_GroundTruth.zip")
    # get_data("https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3B_Test_GroundTruth.zip")
    