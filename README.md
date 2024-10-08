# Dynamic Histogram Algorithm for Time Series Forecasting

## Overview

I implement a dynamic histogram algorithm tailored for time series forecasting. This algorithm is based on the method presented by [Mamonov et al.](https://www.sciencedirect.com/science/article/abs/pii/S0169207022000541), whose formulae and direct methods are not publicly disclosed.

## Methodology

At each forecasting period in the time series, the code computes a histogram. 

Instead of naively building a simple histogram with bins $\( B_i \)$, where the bin count is:

$$
h_i = \sum_{j=1}^{n} \mathbf{1}(x_j \in B_i)
$$

where $\( \mathbf{1}(x_j \in B_i) \)$ is an indicator function that equals 1 if $\( x_j \in B_i \)$, and 0 otherwise.

In this dynamic histogram algorithm, the bin counts are computed using:

$$
h_i = \sum_{j=1}^{n} \text{vicinityfunction}(x_j \in B_i, h)
$$


where `vicinity_function(xj, xh)` is a measure between 0.1 and 0.9, proportional to the proximity of a data point $\( x_j \)$ to the forecasting horizon $\( h \)$.

The strength of this method lies in its vicinity function, which considers:

- Month vicinity
- Day of month vicinity
- Day of week vicinity
- Minute vicinity
- Second vicinity
- Week vicinity

(These are all cyclical measures)

and a non-cyclical measure (weights points based on distance from the horizon without considering cycles)

### Intuitive Example


Suppose you want to forecast running shoe demand using sales data {0 , 1, 2}. 

In a naive histogram algorithm, for each forecasting horizon, you create a histogram from past sales. Each time a sales value appears, you increase the count for that value by one. For instance, if there are 100 '0' sales, the bin for '0' gets a count of 100 * 1 .

In my dynamic histogram algorithm, instead of adding one each time, you add a value proportional to the time distance from the forecasting horizon (which I call the vicinity metric). For example, if those 100 '0 sales' are from 10 years ago,their contribution will be minimal, approaching zero, reflecting their decreased relevance compared to more recent sales. Conversely, if 50 '1 sale' occurred in the last 50 days, each '1 sale''s contribution will be higher, close to 1 , making the bin for '1 sale' potentially higher than the bin for '0 sales', even if the number of '1' sales is fewer.


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Example](#examples)
- [Contact](#contact)
 
## Installation

To use this algorithm, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/charbelkhazen/dynamic_histogram_algorithm.git
cd dynamic_histogram_algorithm
pip install -r requirements.txt
``` 
## Usage

The dynamic histogram has the following tunable hyperparameters:
- Weight of month vicinity
- Weight of day of month vicinity
- Weight of day of week vicinity
- Weight of week vicinity
- Weight of minute vicinity
- Weight of hour vicinity
- Number of bins

(Note: The weight of non-cyclical vicinity defaults to 1 minus the sum of other weights, ensuring their sum equals 1.)

### Finding Optimal Hyperparameters

1. **Select an OOS dataset**: Define a list of horizons using `list_horizons(prct)`.
2. **Evaluate performance**: For each set of hyperparameter , evaluate the OOS perfomance by summing the performance of the set of parameters on the list of horizons that define OOS data.
3. **Choose the best model**: Select the model with the best performance based on cross-validation.

Once you've selected the best model you can use it to determine uncertainty for your desired horizon.


## Contact

Email: khazencharbel@gmail.com

LinkedIn: [charbelkhazen](https://www.linkedin.com/in/charbel-khazen-017285203/)

Website: [charbelkhazen](https://charbelkhazen.com/)
