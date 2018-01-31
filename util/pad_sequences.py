import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack

#######################################

'''
Adopted from:
https://github.com/dhpollack/programming_notebooks/blob/master/pytorch_attention_audio.py#L245
'''

def pad_packed_collate(batch):
    """Puts data, and lengths into a packed_padded_sequence then returns
       the packed_padded_sequence and the labels. Set use_lengths to True
       to use this collate function.
       Args:
         batch: (list of tuples) [(audio, target)].
             audio is a FloatTensor
             target is a LongTensor with a length of 8
       Output:
         packed_batch: (PackedSequence), see torch.nn.utils.rnn.pack_padded_sequence
         labels: (Tensor), labels from the file names of the wav.
    """

    if len(batch) == 1:
        inputs, labels = batch[0][0], batch[0][1]
        #inputs = inputs.t()
        lengths = [inputs.size(0)]
        inputs.unsqueeze_(0)
        labels.unsqueeze_(0)
    if len(batch) > 1:
        #for (a,b) in batch:
        #    print(a.shape)
        #    print(b.shape)
        
        inputs, labels, lengths = zip(*[(a, b, a.shape[1]) for (a,b) in sorted(batch, key=lambda x: x[0].shape[1], reverse=True)])
        #print('inputs0:',inputs[0].shape)
        max_len_inp, H, W = inputs[0].shape[1:]
        inputs = [torch.cat((torch.Tensor(inp), torch.zeros(3, max_len_inp - inp.shape[1], H, W)), dim=1) if inp.shape[1] != max_len_inp else torch.Tensor(inp) for inp in inputs]
        max_len_label, H, W = labels[0].shape
        #for label in labels:
        #    print(torch.Tensor(label).size(), torch.zeros(max_len_label - label.shape[0], H, W).size())
        labels = [torch.cat((torch.Tensor(label), torch.zeros(max_len_label - label.shape[0], H, W)), dim=0) if label.shape[0] != max_len_label else torch.Tensor(label) for label in labels]
        #print(len(inputs))
        inputs = torch.stack(inputs, 0)
        labels = torch.stack(labels, 0)
        #print(inputs.size())
        #print(labels.size())
    #packed_batch = pack(Variable(inputs), lengths, batch_first=True)     #Make sure that we don't need this!
    #packed_labels = pack(Variable(labels), lengths, batch_first=True)    #Make sure that we don't need this!
    #print('after',inputs.size())
    #print('after',labels.size())
    return inputs, labels

########################################################################
'''


# Adopted from Felix Kreuk: XXX
# https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/8


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.shape[dim]
    vec = torch.Tensor(vec)
    print(vec.size())
    zeros_pad = torch.zeros(*pad_size)
    zeros_pad = zeros_pad.view(*pad_size, 1, 1)
    print(zeros_pad.size())
    return torch.cat([vec, zeros_pad], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        # pad according to max_len
        batch = map(lambda xy:
                    (pad_tensor(xy[0], pad=max_len, dim=self.dim), xy[1]), batch)
        # stack all
        xs = torch.stack(list(map(lambda x: x[0], batch)), dim=0)
        ys = torch.LongTensor(list(map(lambda x: x[1], batch)))
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)
        
'''