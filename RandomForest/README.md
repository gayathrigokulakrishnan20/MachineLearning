# Random Forest

**Question:** What if one Decision Tree is not enough?

Random Forest builds many Decision Trees and combines their predictions.

Learning-platform example:
- study hours
- completed lessons
- quiz performance
- days since last login

Target:
- 1 = likely to complete
- 0 = at risk

Important parameters:
- `n_estimators`: number of trees
- `max_depth`: maximum depth of each tree
- `criterion`: split-quality measure
