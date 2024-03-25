"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)
            
    
class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)
            
        
class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64], 
                 sparse=False, embedding_sharing=True):

        super().__init__()

        self.embedding_dim = embedding_dim

        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        self.embedding_sharing = embedding_sharing
        if self.embedding_sharing == True:
            self.user_embedding = ScaledEmbedding(num_users, self.embedding_dim, sparse=sparse)
            self.item_embedding = ScaledEmbedding(num_items, self.embedding_dim, sparse=sparse)
        else:
            self.user_embedding_factor = ScaledEmbedding(num_users, self.embedding_dim, sparse=sparse)
            self.item_embedding_factor = ScaledEmbedding(num_items, self.embedding_dim, sparse=sparse)
            self.user_embedding_reg = ScaledEmbedding(num_users, self.embedding_dim, sparse=sparse)
            self.item_embedding_reg = ScaledEmbedding(num_items, self.embedding_dim, sparse=sparse)        
        
        self.user_bias_embedding = ZeroEmbedding(num_users, 1, sparse=sparse)
        self.item_bias_embedding = ZeroEmbedding(num_items, 1, sparse=sparse)

        self.fc_layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.fc_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.fc_layers.append(nn.ReLU())
        self.fc_layers.append(nn.Linear(layer_sizes[-1], 1))
        
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        
    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of 
            shape (batch,). This corresponds to p_ij in the 
            assignment.
        score: tensor
            Tensor of user-item score predictions of shape 
            (batch,). This corresponds to r_ij in the 
            assignment.
        """
        
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        if self.embedding_sharing == True:
            user_ids_emb = self.user_embedding(user_ids)
            item_ids_emb = self.item_embedding(item_ids)
            
            user_bias_emb = self.user_bias_embedding(user_ids).squeeze()
            item_bias_emb = self.item_bias_embedding(item_ids).squeeze()
            
            predictions = (user_ids_emb * item_ids_emb).sum(dim=1) + user_bias_emb + item_bias_emb

            concat_latent = torch.cat([user_ids_emb, item_ids_emb, user_ids_emb * item_ids_emb], dim=1)
            for layer in self.fc_layers:
                concat_latent = layer(concat_latent).squeeze()
            score = concat_latent
        else:
            user_ids_emb_factor = self.user_embedding_factor(user_ids)
            item_ids_emb_factor = self.item_embedding_factor(item_ids)
            user_ids_emb_reg = self.user_embedding_reg(user_ids)
            item_ids_emb_reg = self.item_embedding_reg(item_ids)
            
            user_bias_emb = self.user_bias_embedding(user_ids).squeeze()
            item_bias_emb = self.item_bias_embedding(item_ids).squeeze()
            
            predictions = (user_ids_emb_factor * item_ids_emb_factor).sum(dim=1) + user_bias_emb + item_bias_emb
            
            concat_latent = torch.cat([user_ids_emb_reg, item_ids_emb_reg, user_ids_emb_reg * item_ids_emb_reg], dim=1)
            for layer in self.fc_layers:
                concat_latent = layer(concat_latent).squeeze()
            score = concat_latent
            
        #print("prediction, score, batch size: ",predictions.size(), score.size(), user_ids.size())
        #********************************************************
        #********************************************************
        #********************************************************
        return predictions, score
