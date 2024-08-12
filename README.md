# Dynamic Histogram Algorithm for Time Series Forecasting

## Overview

I implement a dynamic histogram algorithm tailored for time series forecasting. I reproduce the method presented by [Mamonov et al.](https://www.sciencedirect.com/science/article/abs/pii/S0169207022000541), whose formulae and direct methods are not publicly disclosed.

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
(These are all cyclical)
and a non-cyclical measure (weights points based on distance from the horizon without considering cycles)

### Intuitive Example


Suppose you want to forecast running shoe demand using sales data {0 , 1, 2}. 

In a naive histogram algorithm, for each forecasting horizon, you create a histogram from past sales. Each time a sales value appears, you increase the count for that value by one. For instance, if there are 100 '0' sales, the bin for '0' gets a count of 100 * 1 .

In my dynamic histogram algorithm, instead of adding one each time, you add a value proportional to the time distance from the forecasting horizon (which I call the vicinity metric). For example, if those 100 '0 sales' are from 10 years ago,their contribution will be minimal, approaching zero, reflecting their decreased relevance compared to more recent sales. Conversely, if 50 '1 sale' occurred in the last 50 days, each '1 sale''s contribution will be higher, close to 1 , making the bin for '1 sale' potentially higher than the bin for '0 sales', even if the number of '1' sales is fewer.


## Installation

To use this algorithm, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/charbelkhazen/dynamic_histogram_algorithm.git
cd dynamic_histogram_algorithm
pip install -r requirements.txt
