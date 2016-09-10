require 'nn'
require 'cudnn'
require 'nngraph'
require 'model.LSTM_layer'

opt = {}
opt.input_size = 5
opt.output_size = 3
opt.rnn_size = 6
opt.num_layers = 3
opt.dropout = 0
opt.seq_length = 3
opt.is1vsA = true
m = nn.LSTMLayer(opt)
io.read()
print(torch.type(m))
m:cuda()
