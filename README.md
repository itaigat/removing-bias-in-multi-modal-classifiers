# Removing Bias in Multi-modal Classifiers: Regularization by Maximizing Functional Entropies
## Dependencies
The only dependency is [PyTorch](https://pytorch.org/). We tested it with pytroch 1.4 and 1.5, it should work with all of PyTorch versions.
## Adding our regularization to multi-modal problems

Typically, multi-modal training procedure looks like:

```python
import torch 


for image, question, label in loader:
    logits = model(image, question)
    loss = compute_some_loss(logits, label)

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
```

To use our regularization, you should change the training procedure to

 ```python
import torch
from regularization import Perturbation, Regularization, RegParameters


reg_params = RegParameters()

for image, question, label in loader:
    logits = model(image, question)
    loss = compute_some_loss(logits, label)
    
    ####################### Our regularization method #######################

    expanded_logits = Perturbation.get_expanded_logits(logits, reg_params.n_samples)

    inf_image = Perturbation.perturb_tensor(image, reg_params.n_samples)
    inf_question = Perturbation.perturb_tensor(question, reg_params.n_samples)

    inf_output = model(inf_image, inf_question)
    inf_loss = torch.nn.functional.binary_cross_entropy_with_logits(inf_output, expanded_logits)
    
    gradients = torch.autograd.grad(inf_loss, [inf_image, inf_question], create_graph=True)
    grads = [Regularization.get_batch_norm(gradients[k], loss=inf_loss,
                                           estimation=reg_params.estimation) for k in range(2)]

    inf_scores = torch.stack(grads)
    reg_term = Regularization.get_regularization_term(inf_scores, norm=reg_params.norm,
                                                      optim_method=reg_params.optim_method)
    
    loss += reg_params.delta * reg_term

    #########################################################################

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
```

Note, delta is a scalar controlled by the use.
Note, PyTorch does not allow calculating gradients for Long tensors.
If the input to your model is a Long tensor (might happen for text represented by tokens), we recommend using forward hooking for the first embedding layer's output and calculating the information scores for these tensors.  

## VQA-CPv2 SOTA Use-case
We attach a use-case of how to add our regularization to a given model. We add our regularization term to the original git repository of the paper "[Don't Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases](https://arxiv.org/abs/1909.03683)." 
They use a fork of the [bottom-up attention](https://github.com/chrisc36/bottom-up-attention-vqa) repository.

All the code is under the vqa-cp folder.

## Prerequisites
To install requirements:
```setup
pip install -r requirments.txt
```
## Data Setup

All data should be downloaded to a 'data/' directory in the root
directory of this repository.

The easiest way to download the data is to run the provided script
`tools/download.sh` from the repository root. The features are
provided by and downloaded from the original authors'
[repo](https://github.com/peteanderson80/bottom-up-attention). If the
script does not work, it should be easy to examine the script and
modify the steps outlined in it according to your needs. Then run
`tools/process.sh` from the repository root to process the data to the
correct format.

## Training
Run the following command to train the model with our proposed regularization:

We introduce new parameters:
1. lambda (float; default - 0.0) - scaler of the regularization term.  
2. norm (int; default - 2) - which norm to use. 
3. estimation (str; default - 'ent') - whether the regularization term will be entropy-based or variance-based.
4. optim_method (str; default - 'max_ent') - which optimization method to use. In the paper we present only 'max_ent'.
5. n_samples (int; default = 3) - the number of sample used to estimate the expectation. 
6. grad (bool; default = True) - whether to use gradient bound or not. 
  
```training
python main.py --output saved_models --seed 0 --cache_features --eval_each_epoch --inf_lambda 1e-10
```

## Testing
The scores reported by the script are very close (within a hundredth of a percent in my experience) to the results
reported by the official evaluation metric, but can be slightly different because the 
answer normalization process of the official script is not fully accounted for.
To get the official numbers, you can run `python save_predictions.py /path/to/model /path/to/output_file`
and the run the official VQA 2.0 evaluation on the resulting file. It is available under the eval folder. 

## Pre-trained model
Link to pre-trained model: https://gofile.io/d/FbLhKD.

## Results

Comparison between our method to the previous state-of-the-art 

| Method           | Overall | Yes/No | Number | Other  |
|------------------|---------|--------|--------|--------|
| Learned-Mixin +H | 52.013  | 72.580 | 31.117 | 46.968 |
| Ours             | 54.55   | 74.03  | 49.16  | 45.82  |