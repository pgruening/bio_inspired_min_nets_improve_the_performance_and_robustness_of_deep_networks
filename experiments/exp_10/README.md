# Experiment 10: Minimum Blocks 

Look at weight distribution.
Prune Reference -> What happens?
Prune FP-net -> What happens?
Lottery ticket: Theta-0 vs Theta-j

Net -> one-shot lottery ticket -> can FP-nets be reduced more
x-Axis -> (num params, percentage) / y-Axis -> test-error 

Weight Magnitude vs. Average absolute output

Lottery hypothesis with DRs. Did somebody do this already?

## What is the question that needs to be answered (abstract)?
What is the performance of FPBlocks using the minimum function instead of multiplication?

## Why do you want it to be answered (introduction)?

The main reason why we want to employ explicit multiplications is the modelling of AND terms, i.e., "Feature A AND Feature B is present". Apart from multiplications, different operations come to mind, and we've already established good results by using the babylon trick. However, using the minimum of two value is another promising approach to model a simple logical and relation: the min function only yields a high value if both values are high. The function could be advantegous, (i) because it can only change in a linear fashion and should be easire to optimize, and (ii) using the minimum could be faster.


## How do you try to answer it? (What is the experiment?) (experiments/methods)
We repeat Experiment 0: training on Cifar-10 for five runs for networks with a block number of N=3,5,7,9. This time the multiplication operation of the usual FP-net is substituted with a min operation.

Additionally, we expand the analysis with methods that we've established when writing the JOV paper. E.g., the analysis of angle distributions, entropy, and Dying ReLUs.

## What outcome do you predict and why? 
The LS and MobileNet will probably perform better than the All-FP ResNet as they did before for other block types.
In speed comparison the minimum operation should be the quickest.

## Name at least four possible bugs. Did you check them before running?
- the blocks did not change correctly: *checked*

## Hyperparameters (experiments appendix)
We trained using the default parameters as in experiment 0

## Name at least two tests/outcomes that indicate something is wrong. Did any of this happen?
- remarkable better results than before would be souspicious. 
- the All-FP model is perfoming the best 

## What happened? (results)

### Performance on Cifar10

The model performs slightly worse than its multiplication and babylon counterparts. However, it is ~~about ten times faster in the forward propagation progress~~ the fastes FP-block we have. Thus, the min approach is worth investigating further.

<img src=min_val_er.png width=75%>


## Speed Comparison:

Using the Min-operation is the fastes approach. However, it is not faster than a multiplication block by a wide margin (0.77 vs 0.70 milliseconds). The ReLUBabylon non-linearity is just particularly slow. Interestingly, multiplcation FP-blocks do have a strong standard deviation.

<p float="left">
<img src="time_boxplot_non_lin.png", width=48%>
<img src="time_boxplot_entire_block.png", width=48%>
</p>

#### **Entire blocks**

|      |     FP-mult |   FP-ReLUBabylon |      FP-min |
|:-----|------------:|-----------------:|------------:|
| mean | 0.00076679  |      0.000762099 | 0.000696985 |
| std  | 0.000105456 |      3.28314e-05 | 3.35376e-05 |
| min  | 0.000672579 |      0.0007236   | 0.000661373 |
| max  | 0.00121284  |      0.001127    | 0.000991344 |

#### **Non-Linearities**

|      |   ReLUBabylon |   AbsBabylon |     Minimum |   Multiplication |
|:-----|--------------:|-------------:|------------:|-----------------:|
| mean |   0.000115799 |  0.00010432  | 5.75614e-05 |      6.43404e-05 |
| std  |   5.18779e-06 |  3.9283e-06  | 3.15207e-06 |      7.57467e-06 |
| min  |   0.000109911 |  9.75132e-05 | 5.26905e-05 |      5.53131e-05 |
| max  |   0.000229359 |  0.00015521  | 0.000106096 |      0.0001719   |

## Angle Distribution and Entropy

Min-Nets have a higher angles and the entropy reduction is stronger with increasing entropy.

<p float="left">
<img src="database_images/CifarJOVFPNet_N9_s744_entropy_vs_angle.png", width=48%>
<img src="database_images/CifarMinFP-LS_s723_N9_entropy_vs_angle.png", width=48%>
</p>

<p float="left">
<img src="database_images/CifarJOVFPNet_N9_s744_entropy_vs_angle_regression.png", width=48%>
<img src="database_images/CifarMinFP-LS_s723_N9_entropy_vs_angle_regression.png", width=48%>
</p>

For all blocks the angle distribution is shifted to the right. The blocks are more hyperselective. Does this mean we obtain more robust models?

<p float="left">
<img src="database_images/CifarJOVFPNet_N9_s744_angle_dist.png", width=48%>
<img src="database_images/CifarMinFP-LS_s723_N9_angle_dist.png", width=48%>
</p>

## Dying ReLUs

### Do pyramid- and basic blocks have DRs?

<img src="adjusted_param_num/swarm_resnet_unused_params.png" width=60%>

The original ResNet has 864-144 ununsed parameters. Only for deeper models N>3. DRs occur mainly in earlier conv-layers and only at conv1 of the BasicBlock. However, this is the only ReLU directly placed after a convolution

The plot below shows that only FP-blocks (no-matter the non-linearity) and Basic blocks have dying ReLUs.

<img src="dr_eval/scatter_N9.png" width=60%>


### Are there more DRs for upper, dw, or lower?

<img src="adjusted_param_num/boxplot_conv_key_vs_perc_unused.png" width=60%>

Upper and the DW convolutions are almost identical. However, the number of DRs is reduced in the lower linear combination. Deeper blocks have a higher percentage of DRs. However, shallow blocks have a higher percentage than medium deep blocks.

### Does a specific model type contain more DRs?

<img src="adjusted_param_num/boxplot_model_type_vs_perc_unused.png" width=60%>

The linearized models have a higher percentage of DRs. Which is about 0.4% +/- 0.2% of the entire number of parameters. The multiplication network has fewer DRs. The original ResNet has even fewer. However, there are some configurations with up to 0.4%.

### Error vs unused parameters

There is no obvious correlation between the number of unused parameters and the performance of the model. Shallower models have a higher percentage of unused parameters. This is expectable, because the ratio of FP-block parameters to basic parameters is higher.  

<p float="left">
<img src="scatter_perc_unused_vs_min_val_er.png", width=48%>
<img src="scatter_unused_parameters_vs_min_val_er.png", width=48%>
</p>

## Model Robustness

TODO: Run evaluation for compressed images & adversarial attacks.

## How do you proceed? (discussion)


