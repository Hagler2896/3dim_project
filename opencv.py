import serial
import cv2 # opencv 사용
import numpy as np

ser = serial.Serial('/dev/ttyUSB0',9600)
start_cmd = '#'
stop_cmd = '@'
drive_status = '-1'

l_check = 0
r_check = 0
result_tracking_angle = 0
result_direction = 0

def grayscale(img): # 흑백이미지로 변환
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold): # Canny 알고리즘
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size): # 가우시안 필터
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices, color3=(255,255,255), color1=255): # ROI 셋팅

    mask = np.zeros_like(img) # mask = img와 같은 크기의 빈 이미지
    
    if len(img.shape) > 2: # Color 이미지(3채널)라면 :
        color = color3
    else: # 흑백 이미지(1채널)라면 :
        color = color1
        
    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움 
    cv2.fillPoly(mask, vertices, color)
    
    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=5): # 선 그리기
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_fit_line(img, lines, color=[255, 0, 0], thickness=10): # 대표선 그리기
        cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap): # 허프 변환
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #draw_lines(line_img, lines)
    #cv2.imshow('hough',line_img)

    return lines

def weighted_img(img, initial_img, α=1, β=1., λ=0.): # 두 이미지 operlap 하기
    return cv2.addWeighted(initial_img, α, img, β, λ)

def get_fitline(img, f_lines): # 대표선 구하기   
    lines = np.squeeze(f_lines)
    #print(lines.shape)
    lines = lines.reshape(lines.shape[0]*2,2)
    rows,cols = img.shape[:2]
    output = cv2.fitLine(lines,cv2.DIST_L2,0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3]
    x1, y1 = int(((img.shape[0]-1)-y)/vy*vx + x) , img.shape[0]-1
    x2, y2 = int(((img.shape[0]/2+100)-y)/vy*vx + x) , int(img.shape[0]/2+100)
    
    result = [x1,y1,x2,y2]
    return result
    
def get_line(img,f_lines):
    x1 = f_lines[0,0,0]
    y1 = f_lines[0,0,1]
    x2 = f_lines[0,0,2]
    y2 = f_lines[0,0,3]
    x3 = f_lines[1,0,0]
    y3 = f_lines[1,0,1]
    x4 = f_lines[1,0,2]
    y4 = f_lines[1,0,3]
    
    #cv2.line(img,(x1,y1),(x2,y2),(255,0,0),5)
    
    new_x1 = int((x1 + x3) / 2)
    new_y1 = int((y1 + y3) / 2)
    new_x2 = int((x2 + x4) / 2)
    new_y2 = int((y2 + y4) / 2)
    
    result = [new_x1,new_y1,new_x2,new_y2]
    return result
    


def expression(x1,y1,x2,y2,x3,y3,x4,y4):
    z = 0
    if x2 - x1 != 0:
        m_a = (y2 - y1) / (x2 -x1)
        n_a = -((y2 - y1) / (x2 - x1) * x1 ) + y1
        
    if x4 -x3 != 0:
        m_b = (y4 - y3) / (x4 - x3)
        n_b = -((y4 - y3) / (x4 -x3) * x3 ) + y3
    if x2 - x1 == 0 or x4 - x3 == 0:
        return z
    
    x = (n_b - n_a) / (m_a - m_b) 
    y = m_a * ((n_b - n_a) / (m_a - m_b)) + n_a 
    return x,y

cap = cv2.VideoCapture(0) # 이미지 읽기
drive_status = '0'
ser.write(start_cmd.encode())
ser.write(drive_status.encode())
ser.write(stop_cmd.encode())

while(cap.isOpened()):

    ret,image = cap.read()
#image = cv2.imread('112.jpg')

#height, width = image.shape[:2] # 이미지 높이, 너비
    image = cv2.resize(image, dsize=(320, 240), interpolation=cv2.INTER_AREA)
    m_width = 160

    gray_img = grayscale(image) # 흑백이미지로 변환
    
    blur_img = gaussian_blur(gray_img, 3) # Blur 효과
   
    canny_img = canny(blur_img, 70, 210) # Canny edge 알고리즘

#vertices = np.array([[(50,height),(width/2-45, height/2+60), (width/2+45, height/2+60), (width-50,height)]], dtype=np.int32)
    vertices = np.array([[(10,240),(20,120),(300,120),(310,240)]], dtype=np.int32)
    ROI_img = region_of_interest(canny_img, vertices) # ROI 설정

    line_arr = hough_lines(ROI_img, 1, 1 * np.pi/180, 30, 25, 30) # 허프 변환
    #cv2.imshow('dasd',ROI_img)
#lines=cv2.HoughLinesP(canny_img,1,np.pi/180,30,25,30)


    if line_arr is None:
        pass
    #slope_degree=0
    line_arr = np.squeeze(line_arr)
    #print(int(line_arr.size))
    if int(line_arr.size) > 4:
        #print("slope on")
        slope_degree = (np.arctan2(line_arr[:,1] - line_arr[:,3], line_arr[:,0] - line_arr[:,2]) * 180) / np.pi
        #print(slope_degree)
        L_lines, R_lines = line_arr[(slope_degree>0),:], line_arr[(slope_degree<0),:]
    #L_lines, R_lines = line_arr[(slope_degree>0),:], line_arr[(slope_degree<0),:]
    temp = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    L_lines, R_lines = L_lines[:,None], R_lines[:,None]
    
    #print(L_lines)
    #print(R_lines)
    
    if int(L_lines.size) > 4:
        #left_fit_line = get_fitline(image,L_lines)
        left_fit_line = get_line(image,L_lines)
        draw_fit_line(temp, left_fit_line)
        l_check = 1
        #print("aaa")
        
    if int(R_lines.size) > 4:
        #right_fit_line = get_fitline(image,R_lines)
        right_fit_line = get_line(image,R_lines)
        draw_fit_line(temp, right_fit_line)
        r_check = 1
        #print("bbb")
        
    result = weighted_img(temp, image)
    
    if l_check == 1 and r_check == 1:
        vanishing_point = expression(left_fit_line[0],left_fit_line[1],left_fit_line[2],left_fit_line[3],right_fit_line[0],right_fit_line[1],right_fit_line[2],right_fit_line[3])
        if vanishing_point != 0:
            v_x = int(vanishing_point[0])
            v_y = abs(int(vanishing_point[1]))

        if vanishing_point == 0:
            drive_status = '0'
            continue

        #print(v_x)
        #print(v_y)
        #cv2.circle(result, (v_x,30) , 6,(0,0,255),-1)
        #cv2.line(result,(m_width,0),(m_width,300),(255,255,0),5)

        if(v_x > 120 and v_x < 220):
            #print("Forward")
            drive_status = '0'
            
        if v_x > 220:
            #print("Right!!!")
            #print("Left!!!")
            #cv2.circle(result,(140,50), 6,(0,0,255),-1)
            drive_status = '2'

        if(v_x < 120):
            #print("Left!!!")
            #print("Right!!!")
            #cv2.circle(result,(180,50), 6,(0,0,255),-1)
            drive_status = '3'
    
    if l_check == 1 and r_check == 0:
        
        #cv2.line(result,(0,30),(320,30),(255,255,0),5)
        drive_status = '3'
        #print("right")
    if l_check == 0 and r_check == 1:
        #cv2.line(result,(0,30),(320,30),(255,255,0),5)  
        drive_status = '2'
        #print("left")
    
    ser.write(start_cmd.encode())
    ser.write(drive_status.encode())
    ser.write(stop_cmd.encode())
    l_check = 0
    r_check = 0

    '''
    #print(slope_degree)
    left_line_angle = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    right_line_angle = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    l_n = 0
    r_n = 0

    for number in range(len(slope_degree)):
        #print(number)
        if slope_degree[number] < 0:
            left_line_angle[l_n] = slope_degree[number]
            l_n = l_n + 1
            #print("+")
        else:
            if slope_degree[number] == 90:
                continue
            right_line_angle[r_n] = slope_degree[number]
            r_n = r_n + 1

    #print(left_line_angle)
    #print(right_line_angle)

    left_line_average = 0
    right_line_average = 0

    if l_n > 0:
        left_line_average = sum(left_line_angle) / l_n
        l_check = 1

    if r_n > 0:
        right_line_average = sum(right_line_angle) / r_n
        r_check = 1


    print(left_line_average)
    print(right_line_average)

    if (l_check == 1) and (r_check == 1):
        result_tracking_angle = (abs(left_line_average) + abs(right_line_average)) / 2

    elif (l_check == 1) and (r_check == 0):
        result_tracking_angle = left_line_average

    else:
        result_tracking_angle = right_line_average

    print(result_tracking_angle)

    if left_line_average == 0:
        drive_status = '3';
        print("right")
        
    elif ((result_tracking_angle < 122) and (result_tracking_angle > 107)):
        print("Forward")
        drive_status = '0'
        
    elif right_line_average == 0:
        drive_status = '2'
        print("left")
        
    ser.write(start_cmd.encode())
    ser.write(drive_status.encode())
    ser.write(stop_cmd.encode())
    l_check = 0
    r_check = 0
'''
# 수평 기울기 제한
#line_arr = line_arr[np.abs(slope_degree)<165]
#slope_degree = slope_degree[np.abs(slope_degree)<165]
# 수직 기울기 제한
#line_arr = line_arr[np.abs(slope_degree)>95]
#slope_degree = slope_degree[np.abs(slope_degree)>95]
# 필터링된 직선 버리기
#L_lines, R_lines = line_arr[(slope_degree>0),:], line_arr[(slope_degree<0),:]
#temp = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
#L_lines, R_lines = L_lines[:,None], R_lines[:,None]



#print(line_arr)
#print(L_lines)
#print("--------------------")
#print(R_lines)

# 왼쪽, 오른쪽 각각 대표선 구하기
#left_fit_line = get_fitline(image,L_lines)
#right_fit_line = get_fitline(image,R_lines)
# 대표선 그리기
#draw_fit_line(temp, left_fit_line)
#draw_fit_line(temp, right_fit_line)

    #print(left_fit_line[0])
    #print(left_fit_line[1])
    #print(left_fit_line[2])
    #print(left_fit_line[3])

    #print(right_fit_line[0])
    #print(right_fit_line[1])
    #print(right_fit_line[2])
    #print(right_fit_line[3])

#vanishing_point = expression(left_fit_line[0],left_fit_line[1],left_fit_line[2],left_fit_line[3],right_fit_line[0],right_fit_line[1],right_fit_line[2],right_fit_line[3])

    #print(vanishing_point)

#v_x = int(vanishing_point[0])
#v_y = int(vanishing_point[1])

#result = weighted_img(line_arr, image) # 원본 이미지에 검출된 선 overlap

#cv2.circle(result, (v_x,v_y) , 6,(0,0,255),-1)
#cv2.line(result,(m_width,0),(m_width,300),(255,255,0),5)

#if(v_x > m_width):
 #   print("Right!!!")
  #  cv2.circle(result,(1000,50), 6,(0,0,255),-1)

#if(v_x < m_width):
 #   print("Left!!!")
  #  cv2.circle(result,(100,50), 6,(0,0,255),-1)

    cv2.imshow('ROI',ROI_img) # 결과 이미지 출력
    cv2.imshow('result',result)
#cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


