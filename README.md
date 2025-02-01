# Optimizing MLP Hyperparameters with Algorithmic Methods
A small side project where I experimented with optimizing an MLP with one hidden layer (for computing time reasons). The code could be cleaner and more modular, but after some encouragement, I decided to share it anyway. Feel free to play around and improve it!

Please do not read the notebook inside GitHub, some of these outputs were very verbose (it was much easier to scroll quickly through them while editing inside scrollable sections instead of having to suffer the giant walls of text. Sorry).


## Project Overview
This project explores advanced **hyperparameter optimization** techniques applied to a **Multi-Layer Perceptron (MLP) classifier**. The primary objective is to determine the most effective method for optimizing an MLP's classification performance based on its F1 Score only. The second part tries to balance F1 score and training time.

The optimized hyperparameters are the following:
- **Number of neurons** in the hidden layer.
- **Initial Learning rate (`learning_rate_init`)**.
- **Regularization (`alpha`)**.

  
We evaluate and compare three different optimization strategies:

1. **Genetic Algorithm (GA)**
2. **Bayesian Optimization (Optuna)**
3. **AutoML (TPOT - Genetic Algorithm-Based)**

Additionally, **Multi-Objective Optimization (MOO)** was explored using:
- **NSGA-II** (Non-dominated Sorting Genetic Algorithm II)
- **SMPSO** (Speed-constrained Multi-Objective Particle Swarm Optimization)

## Contents
- `Optimization_Test_mlp.ipynb`: Jupyter Notebook containing the entire experiment.



## Methods and Implementation
### Dataset & Preprocessing
- A **synthetic classification dataset** was generated using `sklearn.datasets.make_classification`.
- **20 features** were created, with **5 informative** and **10 redundant** features.
- **Stratified train-test split (70/30)** ensures class balance.
- **Standardization** was applied using `StandardScaler()`.

### Optimization Techniques
#### 1. Genetic Algorithm (GA)
- Uses `DEAP` for evolutionary hyperparameter tuning.
- Implements **early stopping** for efficiency.
- Uses **tournament selection**, **blend crossover**, and **mutation**.
- Its hyperparameters were optimized using **OPTUNA**, which was very computationally intensive, as one might expect.
- Runs for **multiple generations**, evolving towards better models.

#### 2. Bayesian Optimization
- Uses **Gaussian Process Regression** for intelligent sampling.
- Evaluates fewer hyperparameter configurations but **focuses on promising areas**.
- Applies **cross-validation (5-fold)** to prevent overfitting.
- Optimizes the same hyperparameters as GA.

#### 3. TPOT (AutoML with GA-based Search)
- Uses **Genetic Programming** to evolve full ML pipelines.
- Searches a broader hyperparameter space with automated feature engineering.
- Balances model selection and hyperparameter tuning.

#### 4. Multi-Objective Optimization (MOO) - NSGA-II & SMPSO
- Extends GA optimization to **multiple conflicting objectives**:
  - **Maximize F1-score**
  - **Minimize Overfitting**
  - **Minimize Training Time**
- Finds Pareto-optimal solutions balancing these factors.



## Results & Insights
Do note that all algorithms attained very similar results on test data, with differences within a 0.02 range in test F1 scores. However, overfitting levels varied, with TPOT having the highest train-test gap (0.5).

A model with :
fixed_neurons = 1
fixed_alpha = 0.1
fixed_learning_rate = 0.1

Performed with the following results:
Train F1-Score: 0.7035376183358246
Test F1-Score: 0.7058823529411765

Keep this in mind when observing the following results.

### Bayesian Optimization Performed Best Overall
- **Higher sample efficiency** → Needed fewer function evaluations.
- **Minimal overfitting** compared to GA and TPOT.
- **Achieved a balanced performance without excessive training time.**
- Train F1 = 0.83, Test F1 = 0.81 (0.2 difference between train and test)

### TPOT Achieved the Best F1-score (0.82), but Overfitted More
- **Strong generalization**, but **risk of local optima** due to GA's broad exploration.
- **Less efficient than BO** in lower-dimensional hyperparameter search.
- Train F1 = 0.86, Test F1 = 0.81 (0.5 difference between train and test)

### Genetic Algorithm Performed the Worst
- Had a **higher computational cost**, especially since it also needed tuning.
- **Did not generalize as well** as BO or TPOT.
- **Better suited for larger search spaces (e.g., optimizing multiple layers).**
- Train F1 = 0.82, Test F1 = 0.79 (0.3 difference between train and test)

### Key Takeaway: Use GA for High-Dimensional Search Spaces
- If optimizing in a highly dimensional space (i.e. **number of layers AND neurons per layer**), GA scales better than BO.
- In simple hyperparameter tuning, **BO tends to be the better choice**. If the model is lighter than an MLP with one hidden layer, a GA might be a possibility but at this point...Grid search should do the job.
- **Hybrid Approaches (GA+BO)** might be ideal for large-scale optimizations.


### MOO : NSGA-II vs SMPSO
An image is often worth a thousand words...
![image](https://github.com/user-attachments/assets/f61a793c-7a53-4bb2-8f47-fe39c8482c8e)
The NSGA-II provided better-faring results in practice, but with a lower diversity than the SMPSO. If the NSGA-II didn't somehow provide an exceptionally intriguing solution (F1 = ~0.81 and training time = ~0.1 seconds), SMPSO would seem to be more useful. Both seem to have their merits. One could analyze and experiment with this further.

## How to Run the Code
### Setup Environment
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/mlp-optimization.git
   cd mlp-optimization
   ```
2. Run the Jupyter notebook:
   ```bash
   jupyter notebook Optimization_Test_mlp.ipynb
   ```
   
3. Run everything in order and wait for a while. The OPTUNA + GA step alone took me over three hours of waiting. The rest was actually relatively fast.

 

## Future Work & Improvements
- **Test GA on architecture search** (e.g., optimizing both neurons per layer & number of layers).
- **Experiment with Evolutionary Strategies Algorithm (ES)**.
- **Implement Neural Architecture Search (NAS) for full MLP architecture optimization.**

This project highlights the strengths and trade-offs of different hyperparameter optimization techniques. Depending on the complexity of the search space, different approaches should be chosen accordingly.

 

## Credits & Acknowledgments
This project was conducted as a **personal research initiative** by myself (Ricardo Zwein) alone. The idea was to explore different hyperparameter optimization strategies. All code was written and tested independently for learning purposes and fun.

If you have questions or suggestions, feel free to clone this and have fun!

Link to the notebook on Colab, so you don't have to download it to read it comfortably: [https://colab.research.google.com/drive/16L58YZ3M3PBW3qijvXwfHLw8LBFu3G5B?usp=sharing]
