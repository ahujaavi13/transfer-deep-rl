### Transfer Learning using Deep Reinforcement Learning (DQNs)

This is an exploratory project to investigate the usefulness of transfer learning in the context of Deep Reinforcement Learning. We are using OpenAI Gym’s classic control environments as a tool. Intuition is that knowledge acquired in one environment should aid Agents learning in different environment with similar objectives. We are using two related environments requiring agent to acquire similar knowledge to maximize rewards. Specifically, we are using OpenAI Gym’s Acrobot-v1 and MountainCar-v0 environments, both of which need agent to gain knowledge of momentum. We are using DQN, a simple 3-layer fully connected network (for estimating Q-value) and transferring pre-trained weights from MountainCar-v0 to Acrobot-v1. We explore two techniques, information extraction from weights and changes in architecture. We observed that although there are only minute gains in average reward, the training becomes more stable. Training time stability may be crucial for agents acting in real-world and could decrease side-effects of exploration. Thorough intuition, technical approach (including experiment outcome graphs) are explained in the file ```Project.pdf```.

![](https://raw.githubusercontent.com/ahujaavi13/deep-learning/master/train.gif?token=AB4UFVQGIXQIALV5A77YZAS6EDDWK)

#### **Folder Description**
```
|-----transfer-learning  
     |----mountain-acrobot.py       # Setup - Run this file to reproduce results
     |----agent.py                  # Agent class
     |----experience_replay.py      # Experience replay buffer
     |----model.py                  # Neural network for Q-value approximation
     |----config.py                 # Configuration constants
     |----weights.py                # Pre-trained weights               
            |----MountainCar-v0.pth # MountainedCar pre-trained parameters
            |----MountainCar-v0.pth # Acrobot pre-trained parameters                            
```
#### Replicate the results
The project by default will run 7 times, each exploring a different way to transfer knowledge between the two environments. To start the training, run the below script:

```python mountain-acrobot.py``` or ```python3 mountain-acrobot.py```

To add a custom transfer-learning mechanism, update ```initialize_weights.py``` and ```config.py```. You may also have to update ```model.py``` depending on the method of experimentation. I will be exploring a few more complex ideas directed towards separating general knowledge from environment specific information which may include:
1. Using information gates to better control information flow and distill specifics.
2. Instead of initializing, using pointer networks to provide direct path for information where scores are sampled from pre-trained weights.

Notes: 
1. The pre-trained weights in the "weights" folder are over-trained to capture more information. This included using custom reward for MountainCar-v0 as the environment gets solved at score -110. Both the environments (Acrobot-v1 and MountainCar-v0) were trained for 50k epochs.
2. Training time in default setting is 8-10 hours (depends on GPU though).
3. If running on Colab or Jupyter comment out ```env.render()``` in ```mountain-acrobot.py``` as it may cause unexpected issues
