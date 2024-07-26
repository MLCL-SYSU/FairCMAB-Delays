
Source code using *real-world data* for our paper "Merit-based Fair Combinatorial Semi-Bandit with Unrestricted Feedback Delays".



# Dataset
We download conversion log dataset from <http://research.criteo.com/criteo-sponsored-searchconversion-log-dataset> and extract partial data from the dataset to Top_10_products.txt and Top_10_products_successful_conversions.txt.

- Top_10_products: Data of top 10 products with the most available data in the conversion log dataset.
    
    Content of this dataset
    
    Header Information: < Sale >, < time_delay_for_conversion >, < nb_clicks_1week > ,< product_id > ,< product_title >

    Sale: Indicates 1 if conversion occurred and 0 if not.

    Time_delay_for_conversion: This indicates the time between click and conversion. It is -1, when no conversion took place.
Features.

    nb_clicks_1week: Number of clicks the product related advertisement has received in the last 1 week.

    product_id: Unique identifier associated with every product.

    product_title: Hashed title of the product.



- Top_10_products_successful_conversions: Data of successful conversion (a user clicks and purchases) in Top_10_products.txt.

# Repository Structure

- data: Collect data generated during the execution of the algorithms for simulation, including reward regret, fairness regret, etc.
- algorithms: Detailed procedure of *FCUCB-D, FCTS_D, OP-FCUCB-D* and *OP-FCTS-D* algortihms proposed in the paper.
- arm: Define the reward distributions and feedback delay distributions of the arms.
- plot: Load collected data and generate figures in the paper.
- utilities: A collection of helper functions.
  



# Reproducibility
Run main.py to reproduce the experiments on real-world data under different types of delay distributions.