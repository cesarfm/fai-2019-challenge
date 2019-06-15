def set_all_seeds(seed):
    # python RNG
    import random
    random.seed(seed)

    # pytorch RNGs
    import torch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # numpy RNG
    import numpy as np
    np.random.seed(seed)


def show_top_losses_misclassified(interp, k:int, max_len:int=70)->None:
    """
    Create a tabulation showing the first `k` texts in top_losses along with their prediction, actual,loss, and probability of
    actual class. `max_len` is the maximum number of tokens displayed.
    
    With a modification for showing only the misclassified examples.
    """
    import numpy as np
    import pandas as pd
    from IPython.display import display, HTML
    items = []
    tl_val,tl_idx = interp.top_losses()
    for i,idx in enumerate(tl_idx):
        if k <= 0: break
        tx,cl = interp.data.dl(interp.ds_type).dataset[idx]
        cl = cl.data
        classes = interp.data.classes
        label_pred = classes[interp.pred_class[idx]]
        label_actual = classes[cl]
        if label_pred == label_actual: continue
        k -= 1
        txt = ' '.join(tx.text.split(' ')[:max_len]) if max_len is not None else tx.text
        tmp = [txt, f'{label_pred}', f'{label_actual}', f'{interp.losses[idx]:.2f}',
                f'{interp.preds[idx][cl]:.2f}']
        items.append(tmp)
    items = np.array(items)
    names = ['Text', 'Prediction', 'Actual', 'Loss', 'Probability']
    df = pd.DataFrame({n:items[:,i] for i,n in enumerate(names)}, columns=names)
    with pd.option_context('display.max_colwidth', -1):
        display(HTML(df.to_html(index=False)))
