# Deep Reinforcment Learning case study : Unity Banana environment with DQN

### Set-up the environments
Clone this Git:
```
git clone https://github.com/GabrielLinear/RL_Learning.git
```
Set-up your environment like this [GitHub Pages](https://github.com/udacity/Value-based-methods#dependencies).
Previous to the operation ***pip install .*** , you will have to install torch 0.4 then uninstall it and install the torch version you want.

Then, once you have set-up the environement you can use the trained agent you want like the code bellow. Then you will have the choices between the 3 algorithms implemented.
```
python3 Model_Use.py
```

You can re-train the agent with the different algorithms available like this :
```
python3 DQN.py # To train with the DQN algorithm
python3 Dueling_DQN.py # To train with the Dueling DQN algorithm
python3 Double_DQN.py # To train with the Double DQN algorithm
```

### Environment
Banana environement is a state of the art environmenent provided by Unity to train smart Agents. The goal for the single player is to navigate in the environment to collect yellow banana and avoid blue one. Each yellow banana collected return a postive feedback (a reward) that increase our final score. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The complete environment ( position of bananas, distance from bananas...) is not initially expicitely provided. So our smart agent have to guess from the observations that the environment return for each action carried out.

<p align="center">
  <img src="https://github.com/GabrielLinear/RL_Learning/blob/main/Images/Image.gif" />
</p>

### Results of the algorithms


<p align="center">
  <img src= "https://github.com/GabrielLinear/RL_Learning/blob/main/Images/Scores_Banana.png" />
</p>

For more information you can check the [report](https://github.com/GabrielLinear/RL_Learning/blob/main/Report.pdf). 
