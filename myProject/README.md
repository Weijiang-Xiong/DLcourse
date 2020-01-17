### Integrated Perception and Decision for Self-Driving Simulation

This is a self-driving simulation based on the simulator by UdaCity.

#### Setup

First, you may download the simulator [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-windows.zip), you can extract the files anywhere you like.

UdaCity also provides some wonderful tutorials in [this repo](https://github.com/udacity/self-driving-car-sim).

I used Pytorch 1.2 with CUDA 10.


#### Prepare Data

Start the simulator, and choose Training Mode, and then press R and choose where to store the data, you may set it to `myProject/trainData`.

Now you can start control the car with WASD or the direction keys.

Stop after about 5 laps, and the data would be approximately 400M.

Personally I recorded two sets of data, one is "Mild" and the other is "Aggressive".

In the "Mild" set, I run in the midline of the road, and in the "Aggressive" set, I regard this simulator as a race game.

#### Start Training

In `main.py` , the path to the dataset is in the variable `pathSet`, the model will be saved in `./myProject/tempModels/` every 5 epoch.

The value of training loss and validation loss are saved every 100 batches.

#### Choose a Model

You may use  `plot.py` to plot the loss curve of the training process and chose a proper model among the saved models.

#### Test the Model

Copy your model to `DLcourse/`

Start the simulator and choose Autonomous Mode, type this in the terminal in VSCode:

```
python drive.py <YOUR MODEL NAME> <PATH>
```

The image sequence of this autonomous driving will be saved in `PATH`, if you don't want to save it, just ignore the variable.






