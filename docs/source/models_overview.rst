===========================
Spatial Risk Modeling
===========================

This document provides a clear explanation of different models used for **spatial risk assessment**. While the examples focus on deforestation, these same methods apply to **any binary outcome phenomenon** including:

- **Deforestation & forest degradation** risk
- **Forest fire** occurrence probability
- **Flooding** risk zones
- **Disease outbreak** spatial patterns
- **Landslide** susceptibility
- **Species habitat** suitability
- **Urban expansion** patterns
- **Crop yield** failure zones

Model Classification Summary
==============================

All models predict **binary outcomes** (event occurs: yes/no) at the pixel/location level. The models can be categorized into:

- **Supervised Machine Learning Models**: Trained on historical occurrence data with environmental/contextual predictors
- **Unsupervised/Heuristic Models**: Rule-based approaches using spatial patterns

Models Description
==================

1. Moving Window (MW) Model
----------------------------

**Notebook**: ``4.mw_model.ipynb``

**Type**: Unsupervised spatial heuristic model

**Method**:

- Calculates local event rates within moving windows of different sizes (e.g., 5×5, 11×11, 21×21 pixels)
- Uses historical patterns in the neighborhood to predict future risk
- No machine learning training required

**Key Features**:

- Window sizes: Typically 5, 11, and 21 pixels
- Based on spatial proximity assumption: areas near recent events are at higher risk
- Fast computation, no training phase needed

**Output**: Probability/risk map based on neighborhood event density

**Application Examples**:

- **Deforestation**: Areas near recent forest loss
- **Fire risk**: Zones near previous burn scars
- **Flooding**: Areas near historically flooded zones

**When to use**: Quick baseline model, captures spatial clustering of events

2. Generalized Linear Model (GLM)
----------------------------------

**Notebook**: ``5.2.far_glm.ipynb``

**Type**: Supervised binary classification (regression-based)

**Algorithm**: Logistic Regression

**Method**:

- Uses environmental and contextual variables to predict event probability
- Linear combination of features with logistic transformation
- Trained via maximum likelihood estimation

**Features Used** (examples vary by application):

- **Continuous variables** (scaled): altitude, slope, distances to relevant features (roads, rivers, infrastructure, previous events)
- **Categorical variables**: land use, protected status, soil type, administrative units

**Application Examples**:

- **Deforestation**: altitude, slope, distance to roads/rivers/towns/forest edge, protected areas
- **Fire risk**: temperature, humidity, wind speed, vegetation type, distance to settlements
- **Flooding**: elevation, slope, distance to water bodies, soil permeability, drainage density

**Training**:

- Algorithm: ``sklearn.linear_model.LogisticRegression``
- Sample points from event and non-event locations
- Binary target: ``I(event_occurred)`` where 1 = event happened, 0 = no event

**Output**: Probability of event occurrence (0-1 scale, rescaled to 0-65535 for raster storage)

**Advantages**: Fast, interpretable coefficients, understand which factors drive risk

**Limitations**: Assumes linear relationships (in logit space), doesn't model spatial autocorrelation

3. iCAR Model (Intrinsic Conditional Autoregressive)
-----------------------------------------------------

**Notebook**: ``5.3.far_icar.ipynb``

**Type**: Supervised Bayesian spatial classification

**Algorithm**: Bayesian hierarchical model with spatial random effects

**Method**:

- Extends logistic regression by adding spatial random effects
- Explicitly models spatial autocorrelation via neighborhood structure
- Accounts for the fact that nearby locations tend to have similar risk (spatial dependence)

**Features Used**:

- Same environmental predictors as GLM
- **Additional spatial component**: Cell adjacency matrix (spatial neighborhood structure)

**Training**:

- Bayesian inference via MCMC (Markov Chain Monte Carlo) sampling
- Estimates both coefficients (βs) and spatial autocorrelation parameter (ρ)
- Computationally intensive (requires burn-in and sampling iterations)

**Special Features**:

- Spatial random effects smooth predictions across space
- Interpolates spatial correlation parameter (rho) for fine-scale predictions
- Provides uncertainty estimates via posterior distributions

**Output**: Spatially-smoothed probability map with uncertainty estimates

**Advantages**: Accounts for spatial dependence, more realistic for clustered phenomena like fires, diseases, or deforestation

**Limitations**: Computationally expensive, requires careful tuning (MCMC iterations)

**Application Examples**:

- **Deforestation**: Smooth risk transitions accounting for spatial contagion
- **Disease spread**: Model spatial correlation in outbreak patterns
- **Fire risk**: Account for fire spread patterns and neighborhood effects

4. Random Forest (RF)
----------------------

**Notebook**: ``5.4.far_rf.ipynb``

**Type**: Supervised binary classification (ensemble method)

**Algorithm**: Random Forest Classifier

**Method**:

- Ensemble of decision trees trained on bootstrap samples
- Each tree makes predictions, final output is averaged
- Captures complex non-linear relationships and interactions between features

**Features Used**:

- Same predictors as GLM and iCAR (context-dependent)
- Automatically handles feature interactions and non-linear effects
- Can include temporal features, climate variables, socioeconomic data, etc.

**Training**:

- Algorithm: ``sklearn.ensemble.RandomForestClassifier``
- Parameters: number of trees (typically 100), min samples per leaf, max depth
- No explicit spatial modeling (though can include spatial coordinates)

**Output**: Probability of event occurrence (averaged from all trees)

**Advantages**:

- Captures non-linear relationships and complex interactions
- Robust to outliers and missing data
- Feature importance scores show which variables matter most
- Generally high predictive accuracy across diverse applications

**Limitations**:

- "Black box" model (less interpretable than GLM)
- Can overfit if not properly tuned
- Computationally more expensive than GLM

**Application Examples**:

- **Fire risk**: Complex interactions between weather, vegetation, human activity
- **Flooding**: Non-linear relationships between rainfall, topography, land cover
- **Species distribution**: Complex habitat suitability with multiple interacting factors

5. Benchmark/Stratification Model
-----------------------------------

**Notebook**: ``3.benchmark_jnr_model.ipynb``

**Type**: Unsupervised rule-based spatial stratification

**Method**:

- Stratifies landscape based on key risk factors:

  - Distance to relevant features (e.g., forest edge, water bodies, fault lines)
  - Administrative/ecological units (sub-regions, soil types, etc.)

- Assigns historical event rates to each stratum
- Deterministic assignment (no statistical learning)

**Approach**:

1. Identify distance threshold where most events (e.g., 99.5%) occurred
2. Divide landscape into distance bins from key feature
3. Calculate historical event rate for each bin × category combination
4. Apply these rates as vulnerability scores

**Output**: Vulnerability map with risk classes based on historical patterns

**Advantages**: Simple, transparent, auditable, follows established methodologies (e.g., JNR for deforestation)

**Limitations**: Cannot capture complex interactions, assumes future patterns similar to past

**Application Examples**:

- **Deforestation**: Distance to forest edge × jurisdictions (JNR methodology)
- **Fire risk**: Distance to ignition sources × vegetation types
- **Flooding**: Elevation zones × drainage basins
- **Landslides**: Slope classes × geological units

Model Comparison Table
=======================

.. list-table::
   :header-rows: 1
   :widths: 20 15 10 20 15 15

   * - Model
     - Type
     - Supervised?
     - Spatial Modeling
     - Complexity
     - Interpretability
   * - **Moving Window (MW)**
     - Heuristic
     - No
     - Implicit (neighborhood)
     - Low
     - High
   * - **GLM (Logistic)**
     - Regression
     - Yes
     - No
     - Low
     - High
   * - **iCAR**
     - Bayesian Spatial
     - Yes
     - Explicit (CAR structure)
     - High
     - Medium
   * - **Random Forest**
     - Ensemble
     - Yes
     - No
     - Medium
     - Low
   * - **JNR Benchmark**
     - Rule-based
     - No
     - Implicit (distance-based)
     - Low
     - High

Data Requirements for Training vs. Prediction
===============================================

For SUPERVISED models (GLM, iCAR, Random Forest)
--------------------------------------------------

**Training Phase** - Requires BOTH:

- **Y (target)**: Historical deforestation labels (0 = deforested, 1 = remained forest)
- **X (features)**: Environmental and accessibility variables (altitude, slope, distances, protected areas, etc.)

**Prediction Phase** - Requires ONLY:

- **X (features)**: The same predictor variables for new/future areas
- The trained model applies learned relationships to predict Y

For UNSUPERVISED models (Moving Window, JNR Benchmark)
-------------------------------------------------------

**No Training Phase** - They directly compute predictions from:

- **Historical deforestation patterns** (used as input, not as labeled training data)
- **Spatial/distance features** (forest edge distance, jurisdictions)
- No Y/X distinction - they use deforestation history to create risk zones directly

Training Data
=============

All supervised models (GLM, iCAR, RF) use the same training data generated in:

**Notebook**: ``5.1.far_models_sampling.ipynb``

What is Y (Target/Dependent Variable)?
---------------------------------------

The **binary outcome** for each sampled location between two time periods:

**Examples by application**:

- **Deforestation**: 0 = deforested, 1 = remained forest
- **Fire**: 0 = burned, 1 = not burned
- **Flooding**: 0 = flooded, 1 = not flooded
- **Disease**: 0 = outbreak occurred, 1 = no outbreak

Formula in code: ``I(event_occurred)`` where event is 0 (happened) or 1 (didn't happen)

.. note::
   In the deforestation notebooks, this is coded as ``I(1-deforestation)`` where 1 = forest remained

What are X's (Features/Independent Variables)?
-----------------------------------------------

The predictor variables depend on your specific application. Here are examples across different domains:

**For Deforestation Risk:**

- Environmental: ``altitude``, ``slope``, ``soil_type``
- Accessibility: ``dist_roads``, ``dist_rivers``, ``dist_towns``, ``dist_forest_edge``
- Policy: ``protected_areas``, ``indigenous_territories``, ``jurisdiction``

**For Fire Risk:**

- Climate: ``temperature``, ``humidity``, ``wind_speed``, ``precipitation``
- Vegetation: ``vegetation_type``, ``ndvi``, ``fuel_load``, ``canopy_cover``
- Accessibility: ``dist_settlements``, ``dist_roads``, ``dist_previous_fires``
- Temporal: ``season``, ``fire_season_index``

**For Flooding Risk:**

- Topography: ``elevation``, ``slope``, ``aspect``, ``topographic_wetness_index``
- Hydrology: ``dist_rivers``, ``drainage_density``, ``flow_accumulation``
- Land cover: ``imperviousness``, ``land_use``, ``soil_permeability``
- Infrastructure: ``dist_drainage_systems``, ``dams_upstream``

**For Disease Outbreak:**

- Climate: ``temperature``, ``humidity``, ``rainfall``
- Demographics: ``population_density``, ``age_structure``, ``mobility_patterns``
- Infrastructure: ``healthcare_access``, ``sanitation_quality``
- Proximity: ``dist_previous_cases``, ``dist_high_risk_areas``

**Spatial Variables (for iCAR):**

- ``cell``: Spatial cell ID for modeling neighborhood structure
- ``X, Y``: Geographic coordinates

Sampling Strategy
-----------------

- Stratified random sampling from event and non-event locations
- Typically 10,000+ samples (adaptive based on study area and event prevalence)
- Spatial cell IDs (grid cells of ~10×10 km) for accounting spatial autocorrelation
- Balanced or weighted representation of outcome classes

How Training Works
-------------------

**Supervised models learn the relationship:**

.. code-block:: text

   P(event) = f(X1, X2, X3, ..., Xn)

**Examples:**

- ``P(deforestation) = f(altitude, slope, dist_roads, dist_towns, dist_edge, protected_areas, ...)``
- ``P(fire) = f(temperature, humidity, wind_speed, vegetation_type, dist_settlements, ...)``
- ``P(flooding) = f(elevation, slope, dist_rivers, rainfall, land_use, soil_type, ...)``

**Training process:**

1. Sample locations where we KNOW the outcome (Y = event occurred or not)
2. Extract predictor values (X's) at those locations
3. Fit model to learn: Given these X values, what's the probability of the event?
4. Apply learned model to predict event probability for ALL locations using their X values

Model Evaluation
================

**Notebook**: ``6.models_evaluation.ipynb``

All models are compared using validation metrics on coarse grid cells:

**Metrics**:

- **R²**: Explained variance (how well predictions match observations)
- **RMSE**: Root Mean Square Error (average prediction error)
- **wRMSE**: Weighted RMSE (accounts for varying grid cell sizes)
- **MedAE**: Median Absolute Error (robust to outliers)

**Evaluation Periods**:

- **Calibration**: Training period (e.g., 2015-2020)
- **Validation**: Testing period (e.g., 2020-2024)
- **Historical**: Full historical period (e.g., 2015-2024)
- **Forecast**: Future projections using latest data

Which Model to Choose?
========================

Use Moving Window (MW) when
----------------------------

- Quick assessment needed
- Limited computational resources
- Events are highly spatially clustered (fires, deforestation, disease outbreaks)
- Transparency is critical
- Neighborhood effects dominate other factors

Use GLM when
------------

- Need interpretable coefficients (understand which factors increase/decrease risk)
- Want to quantify driver importance and effect sizes
- Computational efficiency is important
- Linear relationships (in logit space) are reasonable
- Regulatory or policy context requires explainability

Use iCAR when
-------------

- Spatial autocorrelation is strong (contagious processes like fires, disease, deforestation)
- Need spatially-smooth predictions without artificial boundaries
- Have computational resources for MCMC sampling
- Want uncertainty quantification and credible intervals
- Spatial spillover effects are important

Use Random Forest when
-----------------------

- Maximum predictive accuracy is priority
- Relationships are complex/non-linear (e.g., climate thresholds, tipping points)
- Feature interactions are important (e.g., temperature × humidity for fire risk)
- Have many predictors and unsure which matter
- Less concerned about interpretability, more about prediction performance

Use Benchmark/Stratification when
----------------------------------

- Following established standards (e.g., JNR for REDD+, official flooding protocols)
- Need simple, auditable, transparent methodology
- Historical patterns are reliable predictors of future
- Administrative or jurisdictional reporting required
- Stakeholder communication and buy-in are critical
- Limited data or technical capacity

References
==========

- **riskmapjnr**: Python package for JNR risk mapping methodology
- **forestatrisk**: Python package for deforestation risk modeling (GLM, iCAR)
- **sklearn**: Scikit-learn for Random Forest implementation

Additional Notes
=================

Prediction Output Format
-------------------------

All models produce raster maps with values 0-65535 representing deforestation probability:

- 0 = No data / non-forest
- 1-65535 = Risk level (rescaled probability)

Spatial Resolution
------------------

- Typically 30m pixels (matching forest cover data)
- Coarse grid evaluation: 300+ pixel cells for validation

Temporal Periods
----------------

- **Calibration**: Model training period
- **Validation**: Independent test period
- **Historical**: Full observed period (for final model)
- **Forecast**: Future projection period
