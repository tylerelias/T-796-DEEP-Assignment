# T-796-DEEP-Assignment
Group Project in T-796-DEEP

The objective of the assignment is to get hands-on experience in developing deep-learning
neural-network models in PyTorch, going through the process of using external (partially
processed) data, preprocess it, and test, train, experiment with, and employ ANN models.


## Description

The requirements more prediction model must meet are listed below (because this is a
proof-of-concept development there are some simplifying assumptions made, to be
addressed later once the best performing model has been chosen):

- Your model is supposed to predict, for the three companies, whether their stock
price will go up or not (note this is a binary decision, as you are not distinguishing
between the price dropping or staying the same).
- You will get a historical day-to-day record of the stock-price development of the
three companies, along with various information (features) that UpUpUp Inc. is
currently using for its predictions. For details, see appendix. Note that you will have
to preprocess the data to be suitable as an input to your model. Feel free to
omit/use any of the provided information as you deem necessary.
- You should build a single model that predicts all three stock-movements.
- Your models will be evaluated by UpUpUp Inc. using a unified Python interface,
provided in a file predictor.py. You need to implement the predict method in there
to suit your model. Make sure to adhere to the specification, otherwise your
submission cannot be evaluated.
- All models should be built in Google CoLab using PyTorch. Please place the names of
the group members as a text field at the top of your notebook.
