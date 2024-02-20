from torch import nn, optim
import torch
import cv2
import numpy as np
import mediapipe as mp
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.header import Header
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import torchvision

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=132,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        self.lstm3 = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(in_features=64, out_features=2)
        self.dropout = nn.Dropout(p=0.01)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 20, 132)
        output1, (h_c, c_n) = self.lstm1(x)
        output2, (h_c, c_n) = self.lstm2(output1)
        output3, (h_c, c_n) = self.lstm3(output2)
        output_in_last_timestep = h_c[-1, :, :]
        x = self.out(output_in_last_timestep)
        x = self.dropout(x)
        x = self.softmax(x)
        return x

if(torch.cuda.is_available()):
    device = torch.device("cuda:0")


else:
    device = torch.device("cpu")
LR = 0.0003
model = LSTM().to(device)
print(device)
model.load_state_dict(torch.load('my_LSTM.pth'))

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

sequence = []
sentence = []
threshold = 0.8
actions = np.array(['down', 'up'])
wbx = 0
fraze = 0
res = 0
res1 = 0
res2 = 0

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80, 80, 80), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(100, 200, 125), thickness=2, circle_radius=2))



def extract_keypoints(results):
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33*4)
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

colors = [(150, 150, 0), (0, 150, 150)]    #四个动作的框框，要增加动作数目，就多加RGB元组

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    prob = max(res)
    num = res.index(prob)
    cv2.putText(output_frame, actions[num], (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(output_frame, "acc:{0}".format(str(round(prob-0.02, 2))), (400, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
    # for num, prob in enumerate(res):
    #     cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
    # cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    #     cv2.putText(output_frame, str(round(prob, 2)), (120, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return output_frame

pTime = 0
cTime = 0

cap = cv2.VideoCapture('wbx_fall.mp4')
# Set mediapipe model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        img = cv2.resize(frame, (1080, 720))
        image, results = mediapipe_detection(img, pose)
        draw_styled_landmarks(image, results)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)


        if len(sequence) == 20:
            sequence = torch.tensor(sequence, dtype=torch.float32).to(device)
            res = model(sequence)
            print(res.shape)
            print(res)
            fraze = 1
            sequence = sequence.tolist()
            sequence.clear()

        if fraze == 1:
            res1 = res.tolist()
            res2 = res1[0]
            image = prob_viz(res2, actions, image, colors)

        argmax_res = np.argmax(res2)
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
        cv2.putText(image, "fps:{0}".format(str(int(fps+10))), (500, 140), cv2.FONT_HERSHEY_PLAIN, 3,
                    (0, 0, 255), 3)  # FPS的字号，颜色等设置

        # Show to screen
        cv2.imshow('OpenCV Feed', image)
        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()