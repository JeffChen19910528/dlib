import dlib
import cv2

# 初始化 dlib 的人脸检测器和关键点检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 下载链接：http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# 从摄像头获取视频流
cap = cv2.VideoCapture(0)  # 使用第一个摄像头，如果有多个摄像头，可以尝试不同的索引值

while True:
    # 读取视频流的一帧
    ret, frame = cap.read()
    if not ret:
        break

    # 将帧转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = detector(gray)

    # 对每张人脸执行操作
    for face in faces:
        # 检测关键点
        landmarks = predictor(gray, face)

        # 绘制人脸框
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 绘制关键点
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

    # 显示结果
    cv2.imshow("Face Landmarks", frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频流并关闭窗口
cap.release()
cv2.destroyAllWindows()
