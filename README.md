# mlp-optimization
A small side project where I experimented with optimizing an MLP with one hidden layer (for computing time reasons). The code could be cleaner and more modular, but after some encouragement, I decided to share it anyway. Feel free to play around and improve it!

Please do not read the notebook inside GitHub, some of these outputs were very verbose (it was much easier to scroll quickly through them while editing inside scrollable sections instead of having to suffer the giant walls of text. Sorry).


## Project Overview
This project explores advanced **hyperparameter optimization** techniques applied to a **Multi-Layer Perceptron (MLP) classifier**. The primary objective is to determine the most effective method for optimizing an MLP's performance while balancing:

- **F1-score** (classification quality)
- **Overfitting mitigation**
- **Computational efficiency**

We evaluate and compare three different optimization strategies:

1. **Genetic Algorithm (GA)**
2. **Bayesian Optimization (Optuna)**
3. **AutoML (TPOT - Genetic Algorithm-Based)**

Additionally, **Multi-Objective Optimization (MOO)** was explored using:
- **NSGA-II** (Non-dominated Sorting Genetic Algorithm II)
- **SMPSO** (Speed-constrained Multi-Objective Particle Swarm Optimization)

## Contents
- `Optimization_Test_mlp.ipynb`: Jupyter Notebook containing the entire experiment.
- `results/`: Directory containing optimization results and logs.
- `plots/`: Visualizations of optimization performance.



## Methods and Implementation
### Dataset & Preprocessing
- A **synthetic classification dataset** was generated using `sklearn.datasets.make_classification`.
- **20 features** were created, with **5 informative** and **10 redundant** features.
- **Stratified train-test split (70/30)** ensures class balance.
- **Standardization** was applied using `StandardScaler()`.

### Optimization Techniques
#### 1. Genetic Algorithm (GA)
- Uses `DEAP` for evolutionary hyperparameter tuning.
- Optimizes:
  - **Number of neurons** in the hidden layer.
  - **Learning rate (`learning_rate_init`)**.
  - **Regularization parameter (`alpha`)**.
- Implements **early stopping** for efficiency.
- Uses **tournament selection**, **blend crossover**, and **mutation**.
- Runs for **multiple generations**, evolving towards better models.

#### 2. Bayesian Optimization (Optuna)
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
### Bayesian Optimization Performed Best Overall
- **Higher sample efficiency** → Needed fewer function evaluations.
- **Minimal overfitting** compared to GA and TPOT.
- **Achieved a balanced performance without excessive training time.**

### TPOT Achieved the Best F1-score (0.82), but Overfitted More
- **Strong generalization**, but **risk of local optima** due to GA's broad exploration.
- **Less efficient than BO** in lower-dimensional hyperparameter search.

### Genetic Algorithm Performed the Worst
- **Required more generations** and had **higher computational cost**.
- **Did not generalize as well** as BO or TPOT.
- **Better suited for larger search spaces (e.g., optimizing multiple layers).**

### Key Takeaway: Use GA for High-Dimensional Search Spaces
- If optimizing **number of layers AND neurons per layer**, GA scales better than BO.
- In simple hyperparameter tuning, **BO is the best choice**.
- **Hybrid Approaches (GA+BO)** might be ideal for large-scale optimizations.



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
   
3. Install dependencies directly from the notebook by running the appropriate cells.

 

## Future Work & Improvements
- **Test GA on architecture search** (e.g., optimizing both neurons per layer & number of layers).
- **Experiment with Hybrid GA-BO approaches**.
- **Implement Neural Architecture Search (NAS) for full MLP architecture optimization.**

This project highlights the strengths and trade-offs of different hyperparameter optimization techniques. Depending on the complexity of the search space, different approaches should be chosen accordingly.

 

## Credits & Acknowledgments
This project was conducted as a **personal research initiative** exploring hyperparameter optimization strategies. All code was written and tested independently for learning and benchmarking purposes.

**Contact:** If you have questions or suggestions, feel free to reach out!

**References:**
- DEAP: https://deap.readthedocs.io/en/master/
- Optuna: https://optuna.org/
- TPOT: https://epistasislab.github.io/tpot/
- Scikit-learn: https://scikit-learn.org/stable/



Happy Optimizing!


