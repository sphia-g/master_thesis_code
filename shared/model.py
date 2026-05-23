"""Neural network model for tumor grade classification."""

from __future__ import annotations

from typing import List, Union

from torch import nn
from .models import (
    MultiLayerPerceptron,
    LogisticRegression,
    DecisionTree,
    RandomForest,
    KNearestNeighbors,
    SupportVectorMachine,
    GaussianMixtureModel,
    MajorityClassifier,
)
    
def create_model(model_type: str, **kwargs) -> nn.Module:
    """Factory function to create PyTorch models.
    
    Parameters
    ----------
    model_type : str
    
    **kwargs : dict
        Model-specific parameters
    
    Returns
    -------
    nn.Module
        Instantiated PyTorch model
    """
    
    if model_type == 'mlp':
        return MultiLayerPerceptron(
            hidden_layer_sizes=kwargs['hidden_layer_sizes'],
            alpha=kwargs['alpha']
        )
    
    elif model_type == 'logistic_regression':
        return LogisticRegression(
            weight_decay=kwargs['weight_decay']
        )
    
    elif model_type == 'decision_tree':
        return DecisionTree(
            depth=kwargs['depth']
        )
        
    elif model_type == 'knn':
        return KNearestNeighbors(
            k=kwargs['k']
        )
        
    elif model_type == 'svm':
        return SupportVectorMachine(
            C=kwargs['C'],
            kernel=kwargs['kernel'],
            gamma=kwargs['gamma']
        )
        
    elif model_type == 'random_forest':
        return RandomForest(
            n_estimators=kwargs['n_estimators'],
            depth=kwargs['depth'],
            max_features=kwargs['max_features'],
            bootstrap=kwargs['bootstrap']
        )
    elif model_type == 'gmm':
        return GaussianMixtureModel(
            n_components_per_class=kwargs['n_components_per_class'],
            covariance_type=kwargs['covariance_type'],
            max_iter=kwargs['max_iter'],
            tol=kwargs['tol'],
            reg_covar=kwargs['reg_covar']
        )
    elif model_type == 'majority':
        return MajorityClassifier()
    elif model_type == 'tabpfn':
        try:
            from .models.tabpfn import TabPFN
        except ImportError as exc:
            raise ImportError(
                "TabPFN is selected as model type, but the 'tabpfn' package is not available. "
                "Install it in the active environment before starting a TabPFN run."
            ) from exc

        return TabPFN(
            batch_size=kwargs['batch_size'],
            ignore_pretraining_limits=kwargs['ignore_pretraining_limits'],
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
        )