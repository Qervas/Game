import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from collections import deque
import random
import pyautogui
import time
import tkinter as tk
import threading
from mss import mss
import math
from torch.cuda.amp import autocast, GradScaler
import os
import keyboard
import threading
import cv2
import numpy as np
from mss import mss
import torch
import torchvision.transforms as transforms
# global variables
running = True
paused = True
model = None
optimizer = None
enemy_locked = False
close_combat_threshold = 100
fps = 40
frame_buffer = deque(maxlen=5)

# set the game window
bounding_box = {'top': 100, 'left': 100, 'width': 2560, 'height': 1600}
sct = mss()

# action mapping
key_map = {
    'up': 'w',
    'down': 's',
    'left': 'a',
    'right': 'd',
    'jump': 'space',
    'dodge': 'c',
    'block': 'v',
    'heal': 'r',
    'light_attack': 'left_click',
    'heavy_attack': 'right_click',
    'special_attack': 'right_click_hold',
    'lock_target': 'middle_click'
}

class FloatingWindow:
    def __init__(self, master):
        self.master = master
        master.title("Control Panel")
        master.attributes('-topmost', True)
        master.overrideredirect(True)
        master.geometry('200x100+50+50')

        self.paused = False
        
        self.start_pause_button = tk.Button(master, text="开始", command=self.toggle_pause)
        self.start_pause_button.pack(pady=10)

        self.stop_button = tk.Button(master, text="停止", command=self.stop)
        self.stop_button.pack(pady=10)
        keyboard.add_hotkey('f9', self.toggle_pause)
        keyboard.add_hotkey('f10', self.stop)

        master.bind('<Button-1>', self.start_move)
        master.bind('<B1-Motion>', self.do_move)

        # add the method to keep the window on top
        self.keep_on_top()

    def keep_on_top(self):
        self.master.attributes('-topmost', True)
        self.master.after(1000, self.keep_on_top)  # refresh the topmost status every second

    def toggle_pause(self):
        global paused
        paused = not paused
        self.paused = paused
        self.start_pause_button.config(text="Continue" if paused else "Pause")
        print("Program paused" if paused else "Program continued")

    def stop(self):
        global running, model, optimizer
        running = False
        if model is not None and optimizer is not None:
            save_model(model, optimizer)
            print("Model saved")
        else:
            print("Model or optimizer not initialized, cannot save.")
        self.master.quit()

    def start_move(self, event):
        self.x = event.x
        self.y = event.y

    def do_move(self, event):
        deltax = event.x - self.x
        deltay = event.y - self.y
        x = self.master.winfo_x() + deltax
        y = self.master.winfo_y() + deltay
        self.master.geometry(f"+{x}+{y}")

class AdvancedQNetwork(nn.Module):
    def __init__(self, action_space):
        super(AdvancedQNetwork, self).__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        resnet = models.resnet18(weights=weights)
        modules = list(resnet.children())[:-1]
        self.features = nn.Sequential(*modules)
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.features[0] = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.fc = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_space)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CombatSystem:
    def __init__(self):
        self.player_health = 100
        self.player_mana = 100
        self.player_stamina = 100
        self.enemy_health = 100
        self.enemy_posture = 0
        self.player_posture = 0
        self.enemy_attacking = False
        self.player_blocking = False
        self.player_dodging = False
        self.player_jumping = False
        self.enemy_locked = False
        self.healing_flasks = 5
        self.combo_counter = 0
        self.motion_detector = MotionDetector()
        self.last_enemy_position = None
        self.special_attack_charge = 0
        self.special_attack_start_time = None

    def update(self, action, frame):
        # detect motion
        motion = self.motion_detector.detect(frame)
        if motion:
            self.last_enemy_position = motion
            self.enemy_attacking = True
        else:
            self.enemy_attacking = False

        # detect status bars
        self.player_health, self.player_mana, self.player_stamina = detect_status_bars(frame)

        # reset some states
        self.player_dodging = False
        self.player_jumping = False
        previous_enemy_health = self.enemy_health

        # update combat states
        if action == 'block':
            self.player_blocking = True
        elif action == 'dodge':
            if self.player_stamina >= 10:
                self.player_dodging = True
                self.player_stamina -= 10
        elif action == 'jump':
            self.player_jumping = True
        elif action == 'lock_target':
            self.enemy_locked = not self.enemy_locked
        elif action == 'heal':
            if self.healing_flasks > 0:
                self.player_health = min(100, self.player_health + 50)
                self.healing_flasks -= 1
        elif action == 'special_attack':
            current_time = time.time()
            if self.special_attack_start_time is None:
                self.special_attack_start_time = current_time
                self.special_attack_charge = 1
            else:
                charge_duration = current_time - self.special_attack_start_time
                if charge_duration <= 3:
                    self.special_attack_charge = 1
                elif charge_duration <= 6:
                    self.special_attack_charge = 2
                else:
                    self.special_attack_charge = 3
        else:
            self.player_blocking = False
            if self.special_attack_charge > 0:
                # execute special attack
                damage = 15 * self.special_attack_charge * (1 + self.combo_counter * 0.1)
                self.enemy_health -= damage
                self.enemy_posture += 30 * self.special_attack_charge
                self.special_attack_charge = 0
                self.special_attack_start_time = None
            else:
                self.special_attack_start_time = None

        # calculate damage and posture
        if self.enemy_attacking:
            if self.player_blocking:
                    # block only works for remote attack
                pass
            elif self.player_dodging:
                # dodge successfully, no damage
                pass
            elif self.player_jumping:
                # jump may avoid some attacks
                self.player_health -= 5
            else:
                self.player_health -= 10
                self.player_posture += 20
            self.combo_counter = 0  # reset combo when attacked
        else:
            # enemy not attacking, increase combo counter
            if action in ['light_attack', 'heavy_attack', 'special_attack']:
                self.combo_counter += 1

        if action == 'light_attack':
            self.enemy_health -= 5 * (1 + self.combo_counter * 0.1)
            self.enemy_posture += 10
        elif action == 'heavy_attack':
            if self.player_stamina >= 20:
                self.enemy_health -= 10 * (1 + self.combo_counter * 0.1)
                self.enemy_posture += 20
                self.player_stamina -= 20

        # reset posture
        if self.player_posture >= 100:
            self.player_posture = 0
            self.player_health -= 20
        if self.enemy_posture >= 100:
            self.enemy_posture = 0
            self.enemy_health -= 20

        # ensure health is not less than 0
        self.player_health = max(0, self.player_health)
        self.enemy_health = max(0, self.enemy_health)

        # calculate the damage of this attack
        damage_dealt = previous_enemy_health - self.enemy_health

        return damage_dealt

    def get_state(self):
        return {
            'player_health': self.player_health,
            'player_mana': self.player_mana,
            'player_stamina': self.player_stamina,
            'enemy_health': self.enemy_health,
            'player_posture': self.player_posture,
            'enemy_posture': self.enemy_posture,
            'enemy_attacking': self.enemy_attacking,
            'player_blocking': self.player_blocking,
            'player_dodging': self.player_dodging,
            'player_jumping': self.player_jumping,
            'enemy_locked': self.enemy_locked,
            'healing_flasks': self.healing_flasks,
            'combo_counter': self.combo_counter,
            'special_attack_charge': self.special_attack_charge
        }

class MotionDetector:
    def __init__(self):
        self.prev_frame = None

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return None

        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.prev_frame = gray

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            return cv2.boundingRect(c)
        return None

def detect_status_bars(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # detect health bar (white)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    health_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # detect mana bar (blue)
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    mana_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # detect stamina bar (brown)
    lower_brown = np.array([10, 100, 20])
    upper_brown = np.array([20, 255, 200])
    stamina_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # calculate the length of the bars
    health = np.sum(health_mask) // 255
    mana = np.sum(mana_mask) // 255
    stamina = np.sum(stamina_mask) // 255
    
    return health, mana, stamina

def preprocess(frame):
    # ensure the image is RGB format
    if frame.shape[2] == 4:  # if there is an alpha channel
        frame = frame[:, :, :3]  # only keep RGB channels
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(frame).unsqueeze(0)

def capture_screen():
    sct_img = sct.grab(bounding_box)
    # convert sct image to numpy array, and ensure it is RGB format
    frame = np.array(sct_img)
    if frame.shape[2] == 4:  # if there is an alpha channel
        frame = frame[:, :, :3]  # only keep RGB channels
    return frame

def perform_action(action, combat_system):
    if action == 'light_attack':
        pyautogui.click(button='left')
    elif action == 'heavy_attack':
        pyautogui.click(button='right')
    elif action == 'special_attack':
        pyautogui.mouseDown(button='right')
        charge_time = random.uniform(0, 6)  # randomly choose the charge time
        time.sleep(charge_time)
        pyautogui.mouseUp(button='right')
    elif action == 'dodge':
        pyautogui.press('c')
    elif action == 'jump':
        pyautogui.press('space')
    elif action == 'block':
        pyautogui.keyDown('v')
        time.sleep(0.1)
        pyautogui.keyUp('v')
    elif action == 'heal':
        pyautogui.press('r')
    elif action == 'lock_target':
        pyautogui.click(button='middle')
    elif action in ['up', 'down', 'left', 'right']:
        pyautogui.keyDown(key_map[action])
        time.sleep(0.1)
        pyautogui.keyUp(key_map[action])
    
    # update combat system and get the damage dealt
    frame = capture_screen()
    damage_dealt = combat_system.update(action, frame)
    
    return damage_dealt


def calculate_reward(combat_system, action, damage_dealt):
    state = combat_system.get_state()
    reward = 0

    if state['enemy_health'] <= 0:
        reward += 1000  # big reward for defeating the enemy
    elif state['player_health'] <= 0:
        reward -= 1000  # big penalty for player death

    # 耐力管理奖励
    if state['player_stamina'] < 20 and action not in ['heavy_attack', 'dodge']:
        reward += 5  # reward for not using heavy attack and dodge when stamina is low

    if action == 'block' and state['enemy_attacking']:
        reward += 10  # reward for successfully blocking
    elif action == 'dodge' and state['enemy_attacking']:
        reward += 15  # reward for successfully dodging
    elif action == 'jump' and state['enemy_attacking']:
        reward += 5  # reward for successfully jumping

    if action in ['light_attack', 'heavy_attack']:
        reward += damage_dealt * 5  # reward for dealing damage
        if not state['enemy_attacking']:
            reward += damage_dealt * 2  # reward for dealing damage when enemy is not attacking

    if action == 'special_attack':
        if state['special_attack_charge'] > 0:
            reward += 10 * state['special_attack_charge']  # reward for charging special attack
    elif state['special_attack_charge'] > 0:
        reward += damage_dealt * 2 * state['special_attack_charge']  # reward for successful special attack

    if action == 'lock_target' and not state['enemy_locked']:
        reward += 5  # reward for successfully locking target

    if action == 'heal':
        if state['healing_flasks'] > 0:
            reward += 20  # reward for successfully using healing flask
        else:
            reward -= 10  # penalty for trying to use a used healing flask

    reward -= (100 - state['player_health']) * 0.5  # reward for health recovery

    # combo reward
    reward += state['combo_counter'] * 2

    return reward

def train(model, memory, optimizer, device):
    if len(memory) < 1000:
        return

    scaler = GradScaler()
    batch = random.sample(memory, 32)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.cat(states).to(device)
    next_states = torch.cat(next_states).to(device)
    actions = torch.tensor(actions).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    with autocast():
        current_q_values = model(states).gather(1, actions.unsqueeze(1))
        next_q_values = model(next_states).max(1)[0].detach()
        expected_q_values = rewards + (0.99 * next_q_values * (1 - dones))

        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

def save_model(model, optimizer, path='model.pth'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_model(model, optimizer, path='model.pth'):
    if not os.path.exists(path):
        print(f"Model file {path} not found. Starting with a new model.")
        return False
    
    try:
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Model loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Starting with a new model.")
        return False
    return True

def main():
    global model, optimizer, running, paused

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdvancedQNetwork(len(key_map)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    memory = deque(maxlen=100000)
    combat_system = CombatSystem()

    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 5000
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

    frame_idx = 0

    if load_model(model, optimizer):
        print("Model loaded.")
    else:
        print("Starting with a new model.")

    while running:
        if not paused:
            frame = capture_screen()
            state = preprocess(frame).to(device)

            epsilon = epsilon_by_frame(frame_idx)
            if random.random() > epsilon:
                with torch.no_grad():
                    action = model(state).max(1)[1].view(1, 1)
            else:
                action = torch.tensor([[random.randrange(len(key_map))]], device=device, dtype=torch.long)

            action_name = list(key_map.keys())[action.item()]
            damage_dealt = perform_action(action_name, combat_system)

            next_frame = capture_screen()
            next_state = preprocess(next_frame).to(device)

            reward = calculate_reward(combat_system, action_name, damage_dealt)
            done = combat_system.get_state()['player_health'] <= 0 or combat_system.get_state()['enemy_health'] <= 0

            memory.append((state, action, reward, next_state, done))

            train(model, memory, optimizer, device)

            if done:
                combat_system = CombatSystem()  # reset combat system

            frame_idx += 1

            if frame_idx % 1000 == 0:
                save_model(model, optimizer)
                print(f"Model saved at frame {frame_idx}")

        else:
            time.sleep(0.1)

def create_floating_window():
    root = tk.Tk()
    FloatingWindow(root)
    root.mainloop()

def initialize_floating_window():
    window_thread = threading.Thread(target=create_floating_window)
    window_thread.daemon = True
    window_thread.start()

if __name__ == "__main__":
    initialize_floating_window()
    main()