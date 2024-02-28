from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import GINConv, GlobalAvgPool, GCNConv

class GIN0(Model):
    """
    Model class that defines structure of a GIN model. Inherits from tensorflow.keras.models
    Model follows structure proposed in "https://arxiv.org/pdf/1810.00826.pdf"

    ...

    Attributes
    ----------
    channels : int
        Number of channels for a GINConv layer
    n_layers : int
        Number of layers in a model
    n_classes : int
        Number of classes in a dataset
    
    Methods
    -------
    call()
        Gives a prediction based on an input data, performing a forward pass on all layers
    """

    def __init__(self, channels, n_layers, n_classes):
        """ Initializes each layer of a model, creating number of GINConv layers equal to n_layers

        Parameters
        ----------
        channels : int
            Number of channels for a GINConv layer
        n_layers : int
            Number of layers in a model
        n_classes : int
            Number of classes in a dataset
        """
        super().__init__()
        self.conv1 = GINConv(channels, epsilon=0, mlp_hidden=[channels, channels])
        self.convs = []
        for _ in range(1, n_layers):
            self.convs.append(GINConv(channels, epsilon=0, mlp_hidden=[channels, channels]))
        self.pool = GlobalAvgPool()
        self.dense1 = Dense(channels, activation="relu")
        self.dropout = Dropout(0.5)
        self.dense2 = Dense(n_classes, activation="softmax")

    def call(self, inputs):
        """ Produces a prediction based on an input. Gets called when model recives an input

        Parameters
        ----------
        inputs : tuple
            Tuple containing node feature matrix, adjacency matrix and batch number i

        Returns
        -------
        dense2(x) : list
            Vector of size (n_classes, 1), with its bigger number being interpreted as a predicted value
        """

        x, a, i = inputs
        x = self.conv1([x, a])
        for conv in self.convs:
            x = conv([x, a])
        x = self.pool([x, i])
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)
    
class GCNGraph(Model):
    """
    Model class that defines structure of a GCN model, but modified to suit graph classification tasks. Inherits from tensorflow.keras.models
    Original model structure proposed in "https://arxiv.org/pdf/1609.02907.pdf"

    ...

    Attributes
    ----------
    gcn_channels : int
        Number of channels for a GCNConv layer
    dense_channels : int
        Number of channels for a Dense layer
    dropout_rate : float
        Dropout rate between GCNConv layers
    layer_activation : str
        Type of activation function used by GCNConv and Dense layers
    out_activation : str
        Type of activation function used by output layer.
    n_classes : int
        Number of classes in a dataset
    
    Methods
    -------
    call()
        Gives a prediction based on an input data, performing a forward pass on all layers
    """

    def __init__(self, gcn_channels, dense_channels, dropout_rate, layer_activation, out_activation, n_classes, **kwargs):
        """ Initializes each layer of a model.

        Parameters
        ----------
        gcn_channels : int
            Number of channels for a GCNConv layer
        dense_channels : int
            Number of channels for a Dense layer
        dropout_rate : float
            Dropout rate between GCNConv layers
        layer_activation : str
            Type of activation function used by GCNConv and Dense layers
        out_activation : str
            Type of activation function used by output layer.
        n_classes : int
            Number of classes in a dataset
        """
        
        super().__init__(**kwargs)
        self.d1 = Dropout(dropout_rate)
        self.GCN1 = GCNConv(gcn_channels, activation=layer_activation, kernel_regularizer=l2(2.5e-4), use_bias=False)
        self.d2 = Dropout(dropout_rate)
        self.GCN2 = GCNConv(gcn_channels, activation=layer_activation, kernel_regularizer=l2(2.5e-4), use_bias=False)
        self.pool1 = GlobalAvgPool()
        self.dns1 = Dense(dense_channels, activation=layer_activation)
        self.dns2 = Dense(dense_channels/2, activation=layer_activation)
        self.out= Dense(n_classes, activation=out_activation)

    def call(self, inputs):
        """ Produces a prediction based on an input. Gets called when model recives an input

        Parameters
        ----------
        inputs : tuple
            Tuple containing node feature matrix, adjacency matrix and/or batch number i

        Returns
        -------
        out(x) : list
            Vector of size (n_classes, 1), with its bigger number being interpreted as a predicted value
        """

        # Allowing model to work with both batch and disjoint loaders
        if len(inputs) == 2:
            x, a = inputs
        else:
            x, a, _ = inputs


        x = self.d1(x)
        x = self.GCN1([x, a])
        x = self.d2(x)
        x = self.GCN2([x, a])
        x = self.pool1(x)
        x = self.dns1(x)
        x = self.dns2(x)

        return self.out(x)