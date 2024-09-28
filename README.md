# Network Flow Classification - Cyber & ML Project 1
By Kamron Cole

### Repo Overview
- `./data`: datasets used to train models
 - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`: CSV of flows and features
- `./model`: This is where trained pytorch models are saved
 - `autoencoder_405.pth`: PyTorch autoencoder model trianed until a low enough loss was achieved (405 epochs)
  - Identical to `autoencoder_old` structure except the encode layer has an addition normalization layer of 8 nodes
  - Trained using `autoencoder.py`
 - `autoencoder_old_272.pth`: An PyTorch autoencoder model trained until a low enough loss was achieved (272 epochs)
  - Activation Function: ELU
  - Loss Function: Mean Squared Loss
  - Optimization Function: Adam - 0.01 learning rate
  - Encoder with layers with the following amount of nodes, in order
   - 77 - 64 - 32 - 24 - 16 - 8
  - Decoder with the same layers in reverse order.
   - 8 - 16 - 24 - 32 - 64 - 77
  - Trained using `autoencoder_old.py`
 - `classifier_old_254.pth`: A PyTorch model trained until a low enough loss was achieved (254 epochs)
  - Activation Function: ReLU, Sigmoid (Output layer)
  - Loss Function: Binary Cross-Entropy Loss
  - Optimization Function: Adam - 0.01 learning rate
  - A Sequential model with layers with the following amount of nodes, in order
   - 8 - 6 - 5 - 4 - 2 - 1
  - Trained using `autoencoder_old.py`
- `.gitignore`: tells git to ignore specific directories and files
- `autoencoder_old.py`: Un-optimized code for training/testing the autoencoder and classifier models using dataloader. Loads and tests saved old model
- `autoencoder.py`: Optimized code for training/testing the autoencoder and classifier models. This Autoencoder has an additional normalization layer between the encoder and decoder. Trains and tests autoencoder models.
- `confusion_matrix.png`: Graph showing the results of testing the old autoencoder and classifier model with a matrix of actual vs predicted values. New autoencoder produces similar results.
- `feature_idx_to_importance_and_label_old.json`: JSON file saved mapping CSV label to feature importance (difference between base mean loss and the mean loss with a specific feature set to 0 across the dataset). Based on autoencoder trained using `autoencoder_old.py`
- `feature_idx_to_importance_and_label.json`: JSON file saved mapping CSV label to feature importance (difference between base mean loss and the mean loss with a specific feature set to 0 across the dataset). Based on autoencoder trained using `autoencoder.py`
- `mlxautoencoder.py`: Testing the performance of using Apple's MLX library instead of PyTorch. This file was for testing and will not work unless the `mlx` Python package is installed and you are using Apple Hardware.
- `model.py`: Unused file meant to provide abstracted model/training for PyTorch models to classes
- `randomforest.py`: Use scikit-learn package to create, train, and test a Random Forest model on the dataset
- `README.md`: This file
- `requirements.txt`: list of python packages required for files in this folder to work

### Resources Used
1. https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
 - Helped a lot with using PyTorch. I've tried working with TensorFlow before and I had a lot of issues, plus I wanted to work locally on my Macbook, wich tensorflow scarcely supports.
 - This also made me realize some differences though in distinct differences with AI/ML models and their structures. PyTorch does not support RNN/RandomForest like models/algorithms on its own.
2. https://b-nova.com/en/home/content/anomaly-detection-with-random-forest-and-pytorch/
 - Helped a lot with implementing a Random Forest and AutoEncoding models while also helping me understand the concept of the models themselves
3. https://mycourses.rit.edu/d2l/le/content/1105209/viewContent/10136765/View
 - Paper provided for the assignment containing features and algorithm training/testing statistics

# How To Run
### Installing Dependencies
I have confirmed that this code works with `Python 3.12.5` and `Python 3.11.9`. I have also testing my code on the GPU servers.
After installing all the packages in the `requirements.txt` each relevant file should work as is. 

If you dont use `uv` I highly recommend it as a replacement for `pip`. Rust based python package manager. MUCH faster than default `pip`. For reference it took me <1s to resolve the packages, 3 minutes to download, and 5 seconds to install all the packages in the `requirements.txt` with `uv` (3m 6s total), versus 3 minutes to download and 6 minutes to install the packages using `pip` (9m totoal). `conda` would likely have a similar if not worse performance than `pip`.

I then can install packages in the `requirements.txt` using `uv pip install -r requirements.txt`. The commands for `pip` are similar, just remove the `uv` and/or use `pip3` instead.

### Running Models
`randomforest.py` should run pretty quickly on it's own. It will train the Random Forest on 60% of the data, test on the full dataset and plot a confusion matrix.
`autoencoder.py` will train and test models on optimized code from autoencoder_old.py and then test its accuracy on the full dataset and plot a confusion matrix.
`autoencoder_old.py` will load my pre-trained models and then test its accuracy on the full dataset and plot a confusion matrix.


To run the files, be in the root directory of the repo and pass the respective file names to the `python` or `python3` command, e.g.
`python3 autoencoder.py`

I have trained and saved my models used in `autoencoder_old.py` and `autoencoder.py` to the `./model` directory so that they can be loaded and tested, however, all code used to train is included. If this is not sufficient, you can train the models yourself by uncommenting my calls to the training functions. Note that the loss thresholds are not guarenteed to be reached on every run within 300 epochs. Around epoch 290, or sometimes before, somtime after, loss spikes and decreases much slower. When this happens I simply restart the training. I was able to train both models within 10 minutes with `autoencoder_old.py`.

If you plan on training models to test rather than loading my pre-trained ones, I recommend you don't, but, I have optimized my code to make training as fast and easy as possible. EI, if you're going to train and test the models for grading, use `autoencoder.py`. I dont recommend uncommenting the `find_feature_importance` function. This takes a while and is irrelevant to the training and testing of the model, it is just a metric to compare with the paper's features.

I will note, I cannot seem to replicate the training and testing that resulted in `autoencoder_old_272.pth` and `classifier_old_254.pth` with my optimized code, as, whatever Autoencoder model I end up training to a similar loss of `autoencoder_old_272.pth` results in the Classifier model not training (loss doesn't decrease), but if I load the saved `autoencoder_old_272.pth` and train the classifier with that, it does train. The only explaination I can think of for this occuring is that I got extremely lucky with the order of data, number of epochs, and resulting model weights/parameters such that output of the Autoencoder's encode layer ends up normalized. I suspect this mainly because I was only able to get my Classifier model to train with an Autoencoder model that was trained on the optimized code after adding a normalization layer that isn't present in the old models. However, `autoencoder_old.py` consistently produces Autodencoder and Classifier models where their loss decreases, so this might be a property of Dataloader somehow, though I dont know why this would be the case.

With this, follow the instructions in the main functions to use the GPU instead of the CPU.

Note: in `autoencoder.py` I increased the loss threshold to make a successful training more likely.
Also Note: it seems the initialized state of `Classifier` greatly impacts whether it trains or not. If accuracy is not decreasing, re-run. In my testing, this fixed it.

# The Outlook
The outlook for this assignment was to create a model that had similar performances to the ones tested in the provided paper.
We were to choose a day of network data to train our model on and then use their 4 features as input for our model and get similar performance to their. The paper highly recommended a model using the Random Forest (RF) model as it was easily the fastest to train with an accuracy tied for first. However, it was noted a Random Forest implementation was not sufficient for this assignment and we were to use Neural Networks.

I aimed to implement the Random Forest model and a Neural Network model to investigate if the 4 features the paper chose were indeed the most important.
I chose to work on Friday's DDoS flow data as it and DoS intuitively seem to be the easiest to detect from flow data.
All was run/trained locally on my Macbook Pro M3 Max.

# The Data
First, the data. The paper says the prosessing of the raw PCAP data, converted it into flows, each having 80 features. I Used the `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` from the `TrafficLabeling` folder (Not the `MachineLearningCVE`). This provided 250,000 lines of flow data to train on, which I generally give a random 60/40 split; 60% for training, 40% for testing.

When loading the data, I explicitly exclude the Flow ID, Source IP, Source Port, Destination IP, Destination Port, Protocol, and Timestamp columns. I do this to prevent an overfitting of my model to the training data, since a network can have a wide variety of incoming/outgoing IPs, port numbers, timestamps, and the like, that aren't inherently important to detecting if a DDoS is occuring.

# The Random Forest
implemented in `randomforest.py`

Using the 4 features said to be the most important in the paper and using its recommendation to use the RF (Random Forest) algorithm, it uses `scikit-learn` package to create, train, and test the Random Forest model.

The paper reports Random Forest took 74.39 seconds to train and test and resulted in a 0.98 precision, 0.97 recall, and a 0.97 F1 score.

My model is implemented much like theirs using scikit-learn but they do not specify number of n_estimators or random_state. Using 120 n_estimates and ~42% of that (50) as random_state I was able to achieve 0.99 precision, 0.99 recall, and 0.99 F1 score, training on 60% of the data and testing on the remaining 40%. For me, training and testing, takes on average 9-10 seconds.

The improvements in my model are likely due to a few factors. It is likely I may be using more data to train my model, it is also likely that I have more n_estimators and random_state resulting in better performance. If I am using more data to train, my random forest could potentially be overfit to the data.

The differences in execution time are likely just differences in hardware. I am using a M3 Max Macbook Pro.

# The Autoencoder
implemented in `autoencoder.py` and `autoencoder_old.py`

The paper didn't go into much detail with the method used to select the 4 best features for each attack type, so I wanted to find this on my own. The type of model generally used for this is called an AutoEncoder. Each forward step works in two passes, one to an encoder, and another to a decoder which is a mirror of the encoder's structure. The encoder usually takes some number of inputs, and gradually decreases this in subsequent layers to a bottleneck, which is then passed to the decoder. When training this model, then, the loss repressents the ability of the model to reconstruct the initial inputs after being compressed to the bottleneck, meaning the encoder is trained to make it's bottleneck layer some combined representation of the inputs, and is, in a sense, a form of feature extraction. From this, we can then determine the importance of each feature by checking how much impact it has on the loss after training.

When training, I used a rather large but arbitrary batch size of 2560 and a learning rate of 0.01.

For the activation function of each layer I used ELU. I was made aware of it due to the autoencoder reference code I was using and considered changing it, however, ELU allows for/generates negative values as well as positive ones which may be useful when extracting features as some input features may be reduntant or have counter intuative effects on the ability to reconstruct the input data versus ReLU which would treat these values as 0. Essentially, I think ELU allows the autoencoder to be more expressive and therefore accurate, but, I didn't try using other activation functions for the autoencoder so.

With this method, my model seems to have determined the following 8 features to be the most important, in order from most to least important:
1. Flow Bytes/s
2. Idle Max
3. Idle Mean
4. Flow IAT Max
5. Fwd IAT Max
6. Idle Min
7. Flow Duration
8. Fwd IAT Total

The only feature this shares with the 4 from the paper is Flow Duration, and even then it's not in the top 4. This could potentially be due to my model's bottle neck layer being 8 nodes instead of 4, but even then I would expect Flow IAT Std, Bwd Packet Length Std, or Average Package Size to be at least in the top 8 if they were the most influential features.

For perspective, the paper's top 4 features are listed 7th (Flow Duration), 13th (Flow IAT Std), 24th (Bwd Packet Length Std), and 45th (Average Packet Size) when ordered by importance.

Now that I have a trained Autoencoder model, I can extract the encoding layer to compress the 77 inputs into 8. I can then train a separate model meant to classify flows as DDoS or BENIGN based on the 8 extracted features from the autoencoder.

These models I trained until some arbitrarily low loss was achieved. For the autoencoder this was less than 0.000014, and for the classifier, less than 0.0045.

For the classifier, I chose to use a reverse pyramid structure, each layer using ReLU as its activation function, until the last layer of a single node which uses the Signmoid Activation Function. This allows me to treat the output as a probability of an input flow being DDoS or BENIGN.

Using the autoencoder model and classifier model in tandem, and rounding the output of the classifier model results in an even better results than the Random Forest model, result in 1.00 precision, recall, and f1 score. This, of course, is rounded but still shows more accururacy with almost 2 times less false positives and false negatives than the Random Forest model.

The main downside of this is, of course, the time it took to train the autoencoder and the classifier as both required over 250 epochs (272 and 254 respectively) to reach a desired loss value. This itself took about 10 minutes and, using the relative time difference for running the random forest model, would have taken the researchers over an hour. However, less false positives and negatives are highly advantagous when implementing a model like this. It is also possible to adjust the threshold which will adjust the ratio of false negatives to false positives to potentially get a more desirable result.

Another use for an Autoencoder here could be to train the model solely on BENIGN flow data. The purpose here is to "overfit" the model to benign data so that when passed some anomaly flow (like a DDoS flow), the loss calculation should be noticably different. This would likely require more BENIGN data though, which I could get from the other day's CSVs. I attempted such approach but later replaced it for the current one using half of the autoencoder model and a separate classifier model.

# Addendum
I initially typed most of this after training and saving my model(s) using `autoencoder_old.py` before trying my code on the GPU server. I then decided to optimize my code to make training much faster for your convenience if you needed to do that for grading.

As you might have read, this came with a lot of unforseen challenges, but did result in something interesting. Autoencoders trained on the optimized code resulted in slightly different importance for each feature with the top 8 being
1. Flow Bytes/s
2. Idle Max
3. Flow IAT Max
4. Idle Mean
5. Fwd IAT Max
6. Fwd IAT Total
7. Flow Duration
8. Idle Min

and the paper's suggested being 7th (Flow Duration), 12th (Flow IAT Std), 23rd (Bwd Packet Length Std), and 40th (Average Packet Size)