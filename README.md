# MonkeyNet
Classifies different species of monkeys, with deep convolutional neural network. For only 30 epoch I achieve 68% accuracy with a custom designed network architecture.

## Dataset
Let's talk about how you can download the data, first you can download the data from [here](https://www.kaggle.com/slothkong/10-monkey-species)>.

or if you are using google colab or any Jupyter Notebook environment you can run the following command in the cell:
```python
import os
os.environ['KAGGLE_USERNAME'] = "theroyakash"      # Change to your username
os.environ['KAGGLE_KEY'] = "################CONFIDENTIAL################"
```
You can find your kaggle key from account settings by downloading a JSON file. Now once this set, download the data using the following command in a cell.

```python
!kaggle datasets download -d slothkong/10-monkey-species
```
Or you can also use the `terminal`. ZSH or Bash will do. `eff` the windows. Please don't use windows for your DNN training.

## Accuracy
![Imgur1](https://i.imgur.com/LL2dVgg.png)

## Loss
![Imgur2](https://i.imgur.com/STDAcCf.png)

## Architecture
Here is the model architecture. It's 23 layer deep neural network. The data input is 200, 200, 3 color channel. The data flows through the network like this:
![MonkeyNet](https://i.imgur.com/PTR6mw7.png)
