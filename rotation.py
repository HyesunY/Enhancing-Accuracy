import os
'''
pip install cv2 오류나면 opencv-python 설치
pip install opencv-python
'''
import cv2
import numpy as np

# for i in range(1,19,1):
filename = "3.jpg"

# 데이터 전처리를 위한 함수 정의
def preprocess_image(image):
    # 이미지 회전 (90도 반시계 방향으로 회전)
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


    # 이미지 뒤집기 (수평으로 뒤집기)
    flipped_image = cv2.flip(image, 1)

    # 추가적인 전처리 작업 수행 (예: 크기 조절, 색상 변환 등)

    return rotated_image, flipped_image

#그레이 스케일로 변환
src = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
#이미지의 크기가 크기 때문에 5분의1로 줄여서 확인
#안 해도 무방할 듯 함
# src = cv2.resize(src , (int(src.shape[1]/2), int(src.shape[0]/5)))
cv2.imshow('gray',src)
k = cv2.waitKey(0)
cv2.destroyAllWindows()


#양 옆의 노이즈를 제거
src = src[0:src.shape[0]-10, 15:src.shape[1]-25]


#영상 이진화
ret , binary = cv2.threshold(src,170,255,cv2.THRESH_BINARY_INV)
cv2.imshow('binary',binary)
k = cv2.waitKey(0)
cv2.destroyAllWindows()

#노이즈를 제거하기 위해 모폴로지 연산 수행
binary = cv2.morphologyEx(binary , cv2.MORPH_OPEN , cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2)), iterations = 2)
cv2.imshow('binary',binary)
k = cv2.waitKey(0)
cv2.destroyAllWindows()

contours , hierarchy = cv2.findContours(binary , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

#외곽선 검출
#이진화 이미지를 color이미지로 복사(확인용)
color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
#초록색으로 외곽선을 그려준다.
cv2.drawContours(color , contours , -1 , (0,255,0),3)


#리스트연산을 위해 초기변수 선언
bR_arr = []
digit_arr = []
digit_arr2 = []
count = 0


#검출한 외곽선에 사각형을 그려서 배열에 추가
for i in range(len(contours)) :
    bin_tmp = binary.copy()
    x,y,w,h = cv2.boundingRect(contours[i])
    bR_arr.append([x,y,w,h])


# x값을 기준으로 배열을 정렬
bR_arr = sorted(bR_arr, key=lambda num: num[0], reverse=False)
print(bR_arr)
for x, y, w, h in bR_arr:
    tmp_y = bin_tmp[y - 2:y + h + 2, x - 2:x + w + 2].shape[0]
    tmp_x = bin_tmp[y - 2:y + h + 2, x - 2:x + w + 2].shape[1]
    if tmp_x and tmp_y > 10:
        count += 1
        cv2.rectangle(color, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 0, 255), 1)
        digit_arr.append(bin_tmp[y - 2:y + h + 2, x - 2:x + w + 2])
        if count == 5: #count는 1~0 데이터의 행 값
            digit_arr2.append(digit_arr)
            digit_arr = []
            count = 0

cv2.imshow('contours', color)
k = cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range(0,len(digit_arr2)):
    for j in range(len(digit_arr2[i])):
        count += 1
        if i == 0:
            #1일 경우 비율 유지를 위해 마스크를 만들어 그위에 얹어줌
            width = digit_arr2[i][j].shape[1]
            height = digit_arr2[i][j].shape[0]
            tmp = (height - width)/2
            mask = np.zeros((height,height))
            mask[0:height,int(tmp):int(tmp)+width] = digit_arr2[i][j]
            digit_arr2[i][j] = cv2.resize(mask,(28,28))
        else:
            digit_arr2[i][j] = cv2.resize(digit_arr2[i][j],(28,28))

        # 데이터 전처리 함수 호출하여 이미지 처리
        rotated_image, flipped_image = preprocess_image(digit_arr2[i][j])

        # 회전된 이미지 저장
        #cv2.imwrite('./90/' + str(i + 1) + '_' + str(j) + '_rotated.png', rotated_image)

        # 뒤집힌 이미지 저장
        #cv2.imwrite('./flip/' + str(i + 1) + '_' + str(j) + '_flipped.png', flipped_image)

        # 원본 이미지 저장
        cv2.imwrite('./imageResource/' + str(i + 1) + '_' + str(j) + '_original.png', digit_arr2[i][j])
        
        
