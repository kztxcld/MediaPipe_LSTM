from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import mediapipe as mp
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.header import Header
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

sequence = []
sentence = []
threshold = 0.8
actions = np.array(['down', 'up'])
wbx = 0
fraze = 0
res = 0
pTime = 0
cTime = 0

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))


def extract_keypoints(results):
    # pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in pose_landmarks]).flatten()
    else:
        pose = np.zeros(33 * 4)
    return pose

def send_mail():
    con = smtplib.SMTP_SSL('smtp.qq.com', 465)
    con.login('3358638539@qq.com', 'irevdttzqnpqdaec')
    msg = MIMEMultipart()
    subject = Header('监控抓拍', 'utf-8').encode()
    msg['subject'] = subject
    msg['From'] = '3358638539@qq.com <3358638539@qq.com>'
    msg['To'] = '2120426083@qq.com'
    text = MIMEText('老人摔倒，请注意！！！', 'plain', 'utf-8')
    image_data = open('down.jpeg', 'rb').read()
    image1 = MIMEImage(image_data)
    image1['Content-Disposition'] = 'attachment; filename="man_down.jpeg"'
    msg.attach(text)
    msg.attach(image1)
    con.sendmail('3358638539@qq.com', '2120426083@qq.com', msg.as_string())
    con.quit()

colors = [(117, 245, 16), (16, 117, 245)]    #四个动作的框框，要增加动作数目，就多加RGB元组

def prob_viz(res, actions, output_frame, colors):
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(30, 132), recurrent_activation='sigmoid'))
model.add(LSTM(128, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0))
model.add(LSTM(64, return_sequences=False, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0))
model.add(Dense(64, activation='tanh'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(actions.shape[0], activation='softmax'))
model = load_model('action2.h5')

cap = cv2.VideoCapture('phc_test.mp4')
# Set mediapipe model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        img = cv2.resize(frame, (480, 480))
        image, results = mediapipe_detection(img, pose)
        draw_styled_landmarks(image, results)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            fraze = 1
            sequence.clear()

        if fraze == 1:
            image = prob_viz(res, actions, image, colors)

        argmax_res = np.argmax(res)
        if argmax_res == 0:
            wbx += 1
        elif argmax_res == 1:
            wbx -= 1

        if wbx == 80:
            cv2.imwrite('down.jpeg', frame)
            # send_mail()
            wbx = 0
        '''
           视频FPS计算
        '''
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, str(int(fps)), (400, 140), cv2.FONT_HERSHEY_PLAIN, 3,
                    (0, 0, 255), 3)  # FPS的字号，颜色等设置

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('OpenCV Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()