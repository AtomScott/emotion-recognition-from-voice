import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.datasets import tuple_dataset
from chainer import training
from chainer.training import extensions

class CNN(Chain):
    def __init__(self):
        super(CNN, self).__init__()
        with self.init_scope():
            self.cn1 = L.Convolution2D(4, 20, 5)
            self.cn2 = L.Convolution2D(20, 50, 5)
            self.fc1 = L.Linear(800, 500)
            self.fc2 = L.Linear(500, 8)

    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.cn1(x)), 2)
        h2 = F.max_pooling_2d(F.relu(self.cn2(h1)), 2)
        h3 = F.dropout(F.relu(self.fc1(h2)))
        return self.fc2(h3)

image_files = 'image.txt'
dataset = chainer.datasets.LabeledImageDataset(image_files)
split_at = int(len(dataset) * 0.8)
train, test = chainer.datasets.split_dataset(dataset, split_at)

gpu_device = 0
epoch = 30
batch_size = 512
frequency = -1

model = L.Classifier(CNN())
chainer.cuda.get_device_from_id(0)
model.to_gpu()

optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

train_iter = chainer.iterators.SerialIterator(train, batch_size)
test_iter = chainer.iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)

updater = training.StandardUpdater(train_iter, optimizer, device=gpu_device)
trainer = training.Trainer(updater, (epoch, 'epoch'))

trainer.extend(extensions.Evaluator(test_iter, model,device=gpu_device))
trainer.extend(extensions.dump_graph('main/loss'))

frequency = epoch if frequency == -1 else max(1, frequency)
trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
trainer.extend(extensions.LogReport())
trainer.extend(
    extensions.PlotReport(['main/loss', 'validation/main/loss'],
                          'epoch', file_name='loss.png'))
trainer.extend(
    extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],
                          'epoch', file_name='accuracy.png'))
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

trainer.run()
