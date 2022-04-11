# Rock-Paper-Scissors-Image-Model
This is an implementation of an interactive Rock-Paper-Scissors game, in which the user can play with the computer using the camera.

# Milestone 1
Teachable Machine was used to capture images in order to create and train a model with four different classes: Rock, Paper, Scissors and Nothing.

# Milestone 2
A conda environment was created and opencv-python, tensorflow, and ipykernel were downloaded. The model was loaded and run on local machine.

model = load_model('/Users/wadirmalik/Desktop/converted_keras/keras_model.h5')
cap = cv2.VideoCapture(0)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

while rps.your_score or rps.computer_score < int(3): 
    ret, frame = cap.read()
    resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
    image_np = np.array(resized_frame)
    normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
    data[0] = normalized_image
    prediction = model.predict(data)
    cv2.imshow('frame', frame)
    
        # Press q to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()


# Milestone 3
Code was written to first choose an option (Rock, Paper, Scissors) at random and then ask the user for an input. If-Else statement was used to identify the winner. A score counter was added to keep and display the score and a 3 second countdown was added. This was wrapped up in relevant functions.

import random
import time

your_score = 0
computer_score = 0

def rps_rules(x,y):
    global your_score
    global computer_score

    if x == y:
        print("game is a tie")
        print(f'{your_score} - {computer_score}')
        return your_score and computer_score
    elif x == "rock" and y == "scissors":
        print ("computer wins")
        computer_score = computer_score +1
        print(f'{your_score} - {computer_score}')
        return your_score and computer_score       
    elif x == "rock" and y == "paper":
        print ("you win")
        your_score = your_score +1
        print(f'{your_score} - {computer_score}')
        return your_score and computer_score
    elif x == "paper" and y == "rock":
        print ("computer wins")
        computer_score = computer_score +1
        print(f'{your_score} - {computer_score}')
        return your_score and computer_score
    elif x == "paper" and y == "scissors":
        print ("you win")
        your_score = your_score +1
        print(f'{your_score} - {computer_score}')
        return your_score and computer_score
    elif x == "scissors" and y == "paper":
        print ("computer wins")
        computer_score = computer_score +1
        print(f'{your_score} - {computer_score}')
        return your_score and computer_score
    elif x == "scissors" and y == "rock":
        print ("you win")
        your_score = your_score +1
        print(f'{your_score} - {computer_score}')
        return your_score and computer_score

game = ['rock', 'paper', 'scissors']
x = random.choice(game)
y = input('Rock, paper, scissors?')
print (x)
print (y)

def countdown(t):
    
    while t > 0:
        print (t)
        t -= 1
        time.sleep(1)
        
t = int(3)

countdown(t)
rps_rules(x,y)

# Milestone 4
The 2 codes were added together to create the final game, the user input was replaced to gain input from the webcam and a break was added to end the game after someone reaches 3 wins.

import cv2
from tensorflow.keras.models import load_model
import numpy as np
import rps
import random

model = load_model('/Users/wadirmalik/Desktop/converted_keras/keras_model.h5')
cap = cv2.VideoCapture(0)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

rps_dict = {0:'scissors', 1:'rock', 2:'paper',3:'nothing' }

t = int(3)
rps.computer_score = 0
rps.your_score = 0

def rps_keys(values):
    return rps_dict[values]

while rps.your_score or rps.computer_score < int(3): 
    ret, frame = cap.read()
    resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
    image_np = np.array(resized_frame)
    normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
    data[0] = normalized_image
    prediction = model.predict(data)
    cv2.imshow('frame', frame)


    
    rps.countdown(t)
    your_choice_keys = np.argmax(prediction)
    your_choice_name = rps_keys(your_choice_keys)
    print(f'you chose: {your_choice_name}')

    game = ['rock', 'paper', 'scissors']
    computer_choice = random.choice(game)
    print(f'computer chose: {computer_choice}')
    rps.rps_rules(computer_choice,your_choice_name)


    # Press q to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
