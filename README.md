# Deep Reinforcment Learning case study : Unity Banana environment with DQN

### Set-up the environments

Pull this Git:
```
```
Set-up your environment like this [GitHub Pages](https://github.com/udacity/Value-based-methods#dependencies).
Then try to activate and use the enrivonment with an agent taking random action as :

If you want to learn agent with the algorithms do :
```
```

If you want to use the pre-trained algorithms for testing do :
```
```

### Environment
Banana environement is a state of the art environmenent provided by Unity to train smart Agents. The goal for the single player is to navigate in the environment to collect yellow banana and avoid blue one. Each yellow banana collected return a postive feedback (a reward) that increase our final score.
The completes environment ( position of bananas, distance from bananas...) is not initially expicitely provided. So our smart agent have to guess from the observations that the environment return for each action carried out.

![This is an image](https://github.com/GabrielLinear/RL_Learning/blob/main/Images/Image.gif)

Thus for an agent A that take an action each time step t at the environment provide some continous states St.

### Results of the algorithms

![This is an image](https://github.com/GabrielLinear/RL_Learning/blob/main/Images/Scores_Banana.png)
