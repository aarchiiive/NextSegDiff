import nibabel as nib

# NIfTI 파일 경로
nii_path = 'amos22/imagesTr/amos_0071.nii.gz'

# NIfTI 파일 열기
nii_image = nib.load(nii_path)

# 데이터 차원 확인
dimensions = nii_image.shape
print(dimensions)
