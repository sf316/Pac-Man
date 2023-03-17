import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        #Return the dot product of the stored weight vector and the given input, x
        return nn.DotProduct(self.get_weights(), x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        #Return 1 if the dot product is non-negative
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        #Return -1 otherwise
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        #Compute size of data set
        size = 0
        batch_size = 1
        for x,y in dataset.iterate_once(batch_size):
            size += 1
        #Repeatedly loop over the data set
        while(True):
            complete_training_accuracy = 0
            #Loop over the data set
            for x,y in dataset.iterate_once(batch_size):
                #Increase the accuracy if the example is correctly classified
                if self.get_prediction(x) == nn.as_scalar(y):
                    complete_training_accuracy += 1
                #Otherwise make update on examples that are misclassified
                else:
                    direction = x
                    multiplier = nn.as_scalar(y)
                    nn.Parameter.update(self.get_weights(), direction, multiplier)
            #Break the loop when 100% training accuracy has been achieved
            if(complete_training_accuracy == size):
                break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        dimensions = 40
        self.w1 = nn.Parameter(1, dimensions)
        self.w2 = nn.Parameter(dimensions, 1)
        self.b1 = nn.Parameter(1, dimensions)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        #Return f(x) = relu(x dot W1 + b1) dot W2 + b2
        x_dot_W1 = nn.Linear(x, self.w1)
        x_dot_W1_plus_b1 = nn.AddBias(x_dot_W1, self.b1)
        relu = nn.ReLU(x_dot_W1_plus_b1)
        relu_dot_W2 = nn.Linear(relu, self.w2)
        return nn.AddBias(relu_dot_W2, self.b2)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        #Return a loss for batch_size * 1 node of prediction given x and target outputs y
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        #print(dataset.x.shape[0])
        x = nn.Constant(dataset.x)
        y = nn.Constant(dataset.y)
        loss = self.get_loss(x, y)
        multiplier = -0.002
        #Perform gradient-based updates, Extracting a Python floating-point number from a loss node and stop when loss <= 0.02
        while(nn.as_scalar(loss) > 0.02):
            #Construct a loss node
            x = nn.Constant(dataset.x)
            y = nn.Constant(dataset.y)
            loss = self.get_loss(x, y)
            #Obtain the gradients of the loss with respect to the parameters
            grad_wrt_w1, grad_wrt_w2, grad_wrt_b1, grad_wrt_b2 = nn.gradients(loss, [self.w1, self.w2, self.b1, self.b2])
            #Update our parameters
            self.w1.update(grad_wrt_w1, multiplier)
            self.w2.update(grad_wrt_w2, multiplier)
            self.b1.update(grad_wrt_b1, multiplier)
            self.b2.update(grad_wrt_b2, multiplier)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        dimensions = 784
        batch_size = 300
        self.w1 = nn.Parameter(dimensions, batch_size)
        self.w2 = nn.Parameter(batch_size, 10)
        self.b1 = nn.Parameter(1, batch_size)
        self.b2 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        #Return f(x) = relu(x dot W1 + b1) dot W2 + b2
        x_dot_W1 = nn.Linear(x, self.w1)
        x_dot_W1_plus_b1 = nn.AddBias(x_dot_W1, self.b1)
        relu = nn.ReLU(x_dot_W1_plus_b1)
        relu_dot_W2 = nn.Linear(relu, self.w2)
        return nn.AddBias(relu_dot_W2, self.b2)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        #Return a loss for batch_size * 10 node of prediction given x and target outputs y
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        #print(dataset.x.shape[0])
        accuracy = 0
        batch_size = 300
        multiplier = -0.5
        #Perform gradient-based updates, stop when accuracy reaches 97.5%
        while accuracy < 0.975:
            #Loop over the dataset
            for x,y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                #Obtain the gradients of the loss with respect to the parameters
                grad_wrt_w1, grad_wrt_w2, grad_wrt_b1, grad_wrt_b2 = nn.gradients(loss, [self.w1, self.w2, self.b1, self.b2])
                #Update our parameters
                self.w1.update(grad_wrt_w1, multiplier)
                self.w2.update(grad_wrt_w2, multiplier)
                self.b1.update(grad_wrt_b1, multiplier)
                self.b2.update(grad_wrt_b2, multiplier)
            #Compute validation accuracy
            accuracy = dataset.get_validation_accuracy()

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        batch_size = 400
        d = 5
        self.w1 = nn.Parameter(self.num_chars, batch_size)
        self.w2 = nn.Parameter(batch_size, batch_size)
        self.w1h = nn.Parameter(batch_size, batch_size)
        self.w2h = nn.Parameter(batch_size, batch_size)
        self.b1 = nn.Parameter(1, batch_size)
        self.b2 = nn.Parameter(1, batch_size)
        self.b1h = nn.Parameter(1, batch_size)
        self.b2h = nn.Parameter(1, batch_size)
        self.w3 = nn.Parameter(batch_size, d)
        self.b3 = nn.Parameter(1, d)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        #z0 = x0 dot W1
        z0 = nn.Linear(xs[0], self.w1)
        z0_plus_b1 = nn.AddBias(z0, self.b1)
        relu = nn.ReLU(z0_plus_b1)
        relu_dot_w2 = nn.Linear(relu, self.w2)
        #h1 = relu(z0 + b1) dot W2 + b2
        h = nn.AddBias(relu_dot_w2, self.b2)
        index = 1
        l = len(xs)
        #For subsequent letters, compute z_i = x_i dot W + h_i dot W_hidden
        while index < l:
            xi_dot_w = nn.Linear(xs[index], self.w1)
            hi_dot_w1h = nn.Linear(h, self.w1h)
            #z_i = x1 dot W1 + h_i dot W_hidden1
            z = nn.Add(xi_dot_w, hi_dot_w1h)
            z_plus_b1h = nn.AddBias(z, self.b1h)
            relu = nn.ReLU(z_plus_b1h)
            relu_dot_w2h = nn.Linear(relu, self.w2h)
            #h_i+1 = relu(z_i + b_hidden1) dot W_hidden2 + b_hidden2
            h = nn.AddBias(relu_dot_w2h, self.b2h)
            index += 1
        relu = nn.ReLU(h)
        relu_dot_w3 = nn.Linear(relu, self.w3)
        #Return f(h,x) = RelU(h_l) dot W3 + b3
        return nn.AddBias(relu_dot_w3, self.b3)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        #Return a loss for batch_size * d (d = 5) node of prediction given xs and target outputs y
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        index = 0
        number_of_epochs = 20
        batch_size = 200
        multiplier = -0.1
        #Take 20 epochs to train
        while index < number_of_epochs:
            #Loop over the dataset
            for x,y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x,y)
                #Obtain the gradients of the loss with respect to the parameters
                grad_wrt_w1, grad_wrt_w2, grad_wrt_w1h, grad_wrt_w2h, grad_wrt_b1, grad_wrt_b2, grad_wrt_b1h, grad_wrt_b2h, grad_wrt_w3, grad_wrt_b3 = nn.gradients(loss, [self.w1, self.w2, self.w1h, self.w2h, self.b1, self.b2, self.b1h, self.b2h, self.w3, self.b3])
                #Update our parameters
                self.w1.update(grad_wrt_w1, multiplier)
                self.w2.update(grad_wrt_w2, multiplier)
                self.w1h.update(grad_wrt_w1h, multiplier)
                self.w2h.update(grad_wrt_w2h, multiplier)
                self.b1.update(grad_wrt_b1, multiplier)
                self.b2.update(grad_wrt_b2, multiplier)
                self.b1h.update(grad_wrt_b1h, multiplier)
                self.b2h.update(grad_wrt_b2h, multiplier)
                self.w3.update(grad_wrt_w3, multiplier)
                self.b3.update(grad_wrt_b3, multiplier)
            index += 1
