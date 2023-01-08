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

### Hyperboost v2
Beats SMAC in 42% cases, ROAR in 58% cases,
Same as `benchmark-RandomForest` but wit higher learning rate and a scaled knowledge uncertainty 
(because the knowledge uncertainty of virtual ensembles is very low compared to real ensembles)
```python
self.catboost = CatBoostRegressor(iterations=100, loss_function='RMSEWithUncertainty', posterior_sampling=True,
                                          verbose=False, random_seed=0, learning_rate=0.3)
```

```python
return pred[:, 0], knowledge ** 0.3
```

### Hyperboost v3
Beats SMAC in 45% cases, ROAR in 68% cases, V2 in 67% cases
```python
self.catboost = CatBoostRegressor(iterations=100, loss_function='RMSEWithUncertainty', posterior_sampling=False,
                                          verbose=False, random_seed=0, learning_rate=0.3)
```

### Bern0.66Ran1Post0
Beats SMAC in 51% cases, ROAR in 61% cases, V2 in 64% cases and V3 in 48% cases.
```python
self.catboost = CatBoostRegressor(iterations=100, loss_function='RMSEWithUncertainty', posterior_sampling=False,
                                          verbose=False, random_seed=0, learning_rate=0.3, bootstrap_type="Bernoulli",
                                          subsample=0.66)
```

### v3Depthwise
Beats SMAC in 45% cases, ROAR in 71% cases, V2 in 58% cases and V3 in 45% cases
```python
self.catboost = CatBoostRegressor(iterations=100, loss_function='RMSEWithUncertainty', posterior_sampling=False,
                                          verbose=False, random_seed=0, learning_rate=0.3, grow_policy="Depthwise")
```

### Hyperboost v4
Beats SMAC in 64.5% cases, ROAR in 74% cases, V2 in 68% cases and V3 in 55% cases
```python
self.catboost = CatBoostRegressor(iterations=100, loss_function="RMSEWithUncertainty", posterior_sampling=False,
                                          verbose=False, random_seed=0, learning_rate=0.5,
                                          )
```

### Hyperboost v5
Beats SMAC in 84% cases, ROAR in 87% cases, V2 in 84% cases, V3 in 74% cases and V4 in 64.5% cases. 
Has best train loss and train ranking so far.
```python
self.catboost = CatBoostRegressor(iterations=100, loss_function="RMSEWithUncertainty", posterior_sampling=False,
                                          verbose=False, random_seed=0, learning_rate=1.0, random_strength=0)
```