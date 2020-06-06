# Running the project
Create a virtual environment for this project and make sure to **activate** it.

After cloning the project install the necessary libraries:

```pip install -r requirements.txt```


Start the program by running:

```python main.py```

Training of the network should start. To visualize the loss
and the accuracy during training run the follow command in
a new terminal:

```tensorboard --logdir=runs```

Follow the link where you should see a visualization
of the loss and accuracy of the network.

After training, accuracy (across all classes) of the network 
is printed out to the console where ```main.py``` was run.
It should be around 90%. 

A file `myfig.png` should be added to the root directory.
This is a picture of predictions for a batch of 20 images.
Green means the prediction was correct and red means it was
wrong; between parenthesis the true label is shown.


