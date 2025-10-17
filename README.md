### Some interesting keyword and their meaning 
- 🌀 1. What is an “epoch”?
    - In DL training, an epoch means one full pass through the entire training dataset.
      - if your dataset has 10,000 images
      - and you batch size is 100
        - then one epoch = 100 steps (because 10,000 % 100 = 100).
    - During each epoch
      - the model sees every training example once
      - updates weights based on all batches
      - ideally gets a little better
        ```
        Example:
        Suppose you train a cat-vs-dog classifier with 1000 images.
        num_images = 1000
        batch_size = 100
        epochs = 5

        - For each epoch:
            - The model goes through all 1000 images
            - Divided into 10 batches of 100 images each
            - Updates weights after each batch
        
        - Over 5 epochs:
            - The model sees each image 5 times total
            - This usually improves learning up to a point
        
        - Important
            - More epochs -> More learning -> risk of overfitting if too many.
            - Too few epochs -> underfitting
        ```
        
- 🧠2. Why do we use epsilon?
    - It usually used in math operations like division or normalization to:
      - avoid division by zero
      - improve numerical stability
      - Prevent NaNs (Not a Number errors)
      ```
      For example, in normalization:
      x_norm = x / sqrt(variance + eps)
      if variance = 0 then, sqrt(0+0) -> 0 division by 0 -> error
      but with eps:
      sqrt(0 + 1e-5) = sqrt(0.00001)
      by division by zero!
      ```
- Summary Table:
| Term         | Meaning                                                   | Example                                     |
| ------------ | --------------------------------------------------------- | ------------------------------------------- |
| `epoch`      | One complete pass over the entire training dataset        | `epochs = 10` → train 10 times over dataset |
| `eps = 1e-5` | A small constant to avoid division by zero or instability | Used in normalization layers, optimizers    |
