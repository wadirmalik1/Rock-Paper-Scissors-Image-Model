# Rock-Paper-Scissors-Image-Model
This is an implementation of an interactive Rock-Paper-Scissors game, in which the user can play with the computer using the camera.

# Milestone 1
Teachable Machine was used to capture images in order to create and train a model with four different classes: Rock, Paper, Scissors and Nothing.

# Milestone 2
A conda environment was created and opencv-python, tensorflow, and ipykernel were downloaded.

# Milestone 3
Code was written to first choose an option (Rock, Paper, Scissors) at random and then ask the user for an input. If-Else statement was used to identify the winner. This was wrapped up in a function.

def rps_game():

    import random
    game = ['rock', 'paper', 'scissors']
    x = random.choice(game)
    print (x)
    y = input('Rock, paper, scissors')
    print (y)

    if x == y:
        print("game is a tie")
    elif x == "rock" and y == "scissors":
        print ("computer wins")
    elif x == "rock" and y == "paper":
        print ("you win")
    elif x == "paper" and y == "rock":
        print ("computer wins")
    elif x == "paper" and y == "scissors":
        print ("you win")
    elif x == "scissors" and y == "paper":
        print ("computer wins")
    elif x == "scissors" and y == "rock":
        print ("you win")
