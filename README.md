# Speech Lab Project - ImageQA using TensorFlow
NTU CSIE
B03902012 Vincent Huang
B03902090 Nick Shao
## Tensorboard usage
### SSH tunneling
Tensorboard is set to run at `localhost:6006` by default.
To launch from remote server, we have to do ssh tunneling.
Here's an example:
```
ssh -L localhost:16006:localhost:6006 b03902012@server.ip
```
`16006` is any port you like.


### Launch tensorboard

#### logdir

Tensorboard will go to the `logdir` specified in the source code to find the graph we want to visualize.
In `train_gensim.py`, the default logdir is set to be `./tensorboard/`. It can be specified by:

```
python train_gensim.py --logdir=path/to/logdir/
```

#### Launch

```
tensorboard --logdir=path/to/logdir/
```

Make sure that the `path/to/logdir/` is consistent to the one specified in the execution of the training process.
That is to say, Even if you didn't specify the logdir explicitly, you have to use `tensorboard --logdir=./tensorboard/` to run properly.
Some useful arguments:
- `--debug`: print debug messages 

Then, we can navigate the *local* web browser to [localhost:6006](http://localhost:6006) to view the TensorBoard.


For more information, tips and debugging, please refer to the [TensorBoard README](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tensorboard/README.md).
