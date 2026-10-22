# Multi-Modal Product Classifier

## Data preparation

I decided to use "subCategory" because it is not as general as "MasterCategory" and working with "articleType" would be too hard.

There were 22 corrupted CSV rows. I excluded them. There was left around 40 thousand rows. For images, 5 rows did not have the corresponding image. I removed those rows too.

I decided to use only categories with over 300 products as I thought less than that would be hard to train and not too useful. The left categories with the counts are:
-   **Topwear:** 15,398
    
      
    
-   **Shoes:** 7,343
    
      
    
-   **Bags:** 3,055
    
      
    
-   **Bottomwear:** 2,693
    
      
    
-   **Watches:** 2,542
    
      
    
-   **Innerwear:** 1,808
    
      
    
-   **Jewellery:** 1,079
    
      
    
-   **Eyewear:** 1,073
    
      
    
-   **Fragrance:** 1,011
    
      
    
-   **Sandal:** 963
    
      
    
-   **Wallets:** 933
    
      
    
-   **Flip Flops:** 913
    
      
    
-   **Belts:** 811
    
      
    
-   **Socks:** 698
    
      
    
-   **Lips:** 527
    
      
    
-   **Dress:** 478
    
      
    
-   **Loungewear and Nightwear:** 470
    
      
    
-   **Saree:** 427
    
      
    
-   **Nails:** 329
    
      
    
-   **Makeup:** 307


I decided to use my local GPU and CPU for this project. Therefore, using all 40 thousand samples would be too hard, and I decided to cut the number to 10 thousand and keep the proportions of which category takes up the percentage of the sample.

## Models

I decided to split 70% train, 15% validation, 15% test set. The splitting happened within each category so there is no problem in the future, such as most of the train set would be Topwear.

I resized the images to 224x224 and used ImageNet to normalize them. Image manipulations, for example, rotation and flipping, were only applied in training. Text properties were set to 64 pixels.


I used ResNet18 for image model, DistilBERT for text model and their 512 and 768 value representations passed to MLP classifier for the hybrid model.


For our hybrid model, instead of using the information from image and text together from the start, we take their separate decisions and create a combined summary to determine the category.

The batch size is 32 for all epochs, and I used AdamW with learning rate 0.001 and weight decay 0.01 on my laptop GPU ( RTX 3060 ).
Macro F1 validation was used to select checkpoints. I did not attempt full fine-tuning but normalized the batches using ResNet.


## Results

These are the notebook’s saved test results for the unweighted runs:

-   **Image-only**
    
      
    -   Accuracy: 0.8613
        
          
        
    -   Macro F1: 0.7062
        
          
        
    -   Weighted F1: 0.8537
        
          
        
-   **Text-only**
    
      
    -   Accuracy: 0.8193
        
          
        
    -   Macro F1: 0.6461
        
          
        
    -   Weighted F1: 0.7817
        
          
        
-   **Image + Text**
    
      
    -   Accuracy: 0.9627
        
          
        
    -   Macro F1: 0.9179
        
          
        
    -   Weighted F1: 0.9628


Macro F1 gives each category equal importance therefore it is the main metric. The bigger categories influenced the score more.

Class weighting, calculated from training labels only, changed validation macro F1 from 0.7380 to 0.7428 for images, 0.6833 to 0.7349 for text, and 0.9261 to 0.8813 for fusion. It helped the individual models in these runs but hurt fusion. Weighted models were not evaluated on the test set.

## Error analysis

The fused model made 56 test errors. The notebook displays five examples.

After checking the errors, I can say that there would be a probability of me also making an error when identifying the category for those products, therefore, I do not think it is a big issue.

## Conclusion and limitations

Fusion performed best in this experiment: test macro F1 improved by 0.2117 over image-only and 0.2718 over text-only, or 21.17 and 27.18 percentage points.

This works for explicitly this setup. Our model has batch normalization and different dropout. Maybe if some fine-tuning was applied, there would be different results.


## BONUS
As a bonus, I tested class weighting to give smaller categories more importance during training. Validation macro F1 increased from 0.7380 to 0.7428 for image-only and from 0.6833 to 0.7349 for text-only, but decreased from 0.9261 to 0.8813 for fusion. This shows that class weighting helped the individual models in these runs, but did not improve every model.
