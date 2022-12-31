# Benchmarks

### benchmark-RandomForest

Initialization:

```python
self.catboost = CatBoostRegressor(iterations=100, loss_function='RMSEWithUncertainty', posterior_sampling=True,
                                          verbose=False, random_seed=0, learning_rate=0.03)
```

Prediction:
```python
pred = self.catboost.predict(X)
preds = self.catboost.virtual_ensembles_predict(X, prediction_type='TotalUncertainty',
                                                        virtual_ensembles_count=10)
mean_preds = preds[:, 0]  # mean values predicted by a virtual ensemble
knowledge = preds[:, 1]  # knowledge uncertainty predicted by a virtual ensemble
data = preds[:, 2]  # average estimated data uncertainty

return pred[:, 0], knowledge
```

### virt20-RandomForest
Same as `benchmark-RandomForest` but with 20 virtual ensembles:
```python
preds = self.catboost.virtual_ensembles_predict(X, prediction_type='TotalUncertainty',
                                                        virtual_ensembles_count=20)
```

### data-RandomForest
Same as `benchmark-RandomForest` but using data as knowledge uncertainty:
```python
return pred[:, 0], data
```

### v2-RandomForest
Same as `benchmark-RandomForest` but wit higher learning rate and a scaled knowledge uncertainty 
(because the knowledge uncertainty of virtual ensembles is very low compared to real ensembles)
```python
self.catboost = CatBoostRegressor(iterations=100, loss_function='RMSEWithUncertainty', posterior_sampling=True,
                                          verbose=False, random_seed=0, learning_rate=0.3)
```

```python
return pred[:, 0], knowledge ** 0.3
```