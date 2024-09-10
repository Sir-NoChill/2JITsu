#!/usr/bin/env python
# coding: utf-8

# See code at https://github.com/google-research/vision_transformer/
# 
# See papers at
# 
# - Vision Transformer: https://arxiv.org/abs/2010.11929
# - MLP-Mixer: https://arxiv.org/abs/2105.01601
# - How to train your ViT: https://arxiv.org/abs/2106.10270
# - When Vision Transformers Outperform ResNets without Pretraining or Strong Data Augmentations: https://arxiv.org/abs/2106.01548
# 
# This Colab allows you to run the [JAX](https://jax.readthedocs.org) implementation of the Vision Transformer.
# 
# If you just want to load a pre-trained checkpoint from a large repository and
# directly use it for inference, you probably want to go [this Colab](https://colab.research.google.com/github/google-research/vision_transformer/blob/main/vit_jax_augreg.ipynb).

# ##### Copyright 2021 Google LLC.

# 

# In[1]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# <a href="https://colab.research.google.com/github/google-research/vision_transformer/blob/main/vit_jax.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ### Setup
# 
# Needs to be executed once in every VM.
# 
# The cell below downloads the code from Github and install necessary dependencies.

# In[2]:


# Clone repository and pull latest changes.
get_ipython().system('[ -d vision_transformer ] || git clone --depth=1 https://github.com/google-research/vision_transformer')
get_ipython().system('cd vision_transformer && git pull')


# In[3]:


# Colab already includes most of the dependencies, so we only install the delta:
#!pip install einops>=0.3.0 ml-collections>=0.1.0 aqtp>=0.2.0 clu>=0.0.3 git+https://github.com/google/flaxformer tensorflow-text>=2.9.0

get_ipython().system('pip install -qr vision_transformer/vit_jax/requirements.txt')
get_ipython().system('pip install gsutil')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install -U "jax[cuda12]"')


# ### Imports

# In[4]:


# Shows all available pre-trained models.
get_ipython().system('gsutil ls -lh gs://vit_models/imagenet*')
get_ipython().system('gsutil ls -lh gs://vit_models/sam')
get_ipython().system('gsutil ls -lh gs://mixer_models/*')


# In[5]:


# Download a pre-trained model.

# Note: you can really choose any of the above, but this Colab has been tested
# with the models of below selection...
model_name = 'ViT-B_32'  #@param ["ViT-B_32", "Mixer-B_16"]

if model_name.startswith('ViT'):
  get_ipython().system('[ -e "$model_name".npz ] || gsutil cp gs://vit_models/imagenet21k/"$model_name".npz .')
if model_name.startswith('Mixer'):
  get_ipython().system('[ -e "$model_name".npz ] || gsutil cp gs://mixer_models/imagenet21k/"$model_name".npz .')

import os
assert os.path.exists(f'{model_name}.npz')


# In[6]:


# Google Colab "TPU" runtimes are configured in "2VM mode", meaning that JAX
# cannot see the TPUs because they're not directly attached. Instead we need to
# setup JAX to communicate with a second machine that has the TPUs attached.
import os
if 'google.colab' in str(get_ipython()) and 'COLAB_TPU_ADDR' in os.environ:
  import jax
  import jax.tools.colab_tpu
  jax.tools.colab_tpu.setup_tpu()
  print('Connected to TPU.')
else:
  print('No TPU detected. Can be changed under "Runtime/Change runtime type".')


# In[7]:


from absl import logging
import flax
import jax
from matplotlib import pyplot as plt
import numpy as np
import optax
import tqdm

logging.set_verbosity(logging.INFO)

# Shows the number of available devices.
# In a CPU/GPU runtime this will be a single device.
# In a TPU runtime this will be 8 cores.
jax.local_devices()


# In[8]:


# Import files from repository.
# Updating the files in the editor on the right will immediately update the
# modules by re-importing them.

import sys
if './vision_transformer' not in sys.path:
  sys.path.append('./vision_transformer')

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from vit_jax import checkpoint
from vit_jax import input_pipeline
from vit_jax import utils
from vit_jax import models
from vit_jax import train
from vit_jax.configs import common as common_config
from vit_jax.configs import models as models_config


# In[9]:


# Helper functions for images.

labelnames = dict(
  # https://www.cs.toronto.edu/~kriz/cifar.html
  cifar10=('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
  # https://www.cs.toronto.edu/~kriz/cifar.html
  cifar100=('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'computer_keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm')
)
def make_label_getter(dataset):
  """Returns a function converting label indices to names."""
  def getter(label):
    if dataset in labelnames:
      return labelnames[dataset][label]
    return f'label={label}'
  return getter

def show_img(img, ax=None, title=None):
  """Shows a single image."""
  if ax is None:
    ax = plt.gca()
  ax.imshow(img[...])
  ax.set_xticks([])
  ax.set_yticks([])
  if title:
    ax.set_title(title)

def show_img_grid(imgs, titles):
  """Shows a grid of images."""
  n = int(np.ceil(len(imgs)**.5))
  _, axs = plt.subplots(n, n, figsize=(3 * n, 3 * n))
  for i, (img, title) in enumerate(zip(imgs, titles)):
    img = (img + 1) / 2  # Denormalize
    show_img(img, axs[i // n][i % n], title)


# ### Load dataset

# In[10]:


dataset = 'cifar10'
batch_size = 512
config = common_config.with_dataset(common_config.get_config(), dataset)
config.batch = batch_size
config.pp.crop = 224


# In[11]:


# For details about setting up datasets, see input_pipeline.py on the right.
ds_train = input_pipeline.get_data_from_tfds(config=config, mode='train')
ds_test = input_pipeline.get_data_from_tfds(config=config, mode='test')
num_classes = input_pipeline.get_dataset_info(dataset, 'train')['num_classes']
del config  # Only needed to instantiate datasets.


# In[12]:


# Fetch a batch of test images for illustration purposes.
batch = next(iter(ds_test.as_numpy_iterator()))
# Note the shape : [num_local_devices, local_batch_size, h, w, c]
batch['image'].shape


# In[13]:


# Show some images with their labels.
images, labels = batch['image'][0][:9], batch['label'][0][:9]
titles = map(make_label_getter(dataset), labels.argmax(axis=1))
show_img_grid(images, titles)


# In[14]:


# Same as above, but with train images.
# Note how images are cropped/scaled differently.
# Check out input_pipeline.get_data() in the editor at your right to see how the
# images are preprocessed differently.
batch = next(iter(ds_train.as_numpy_iterator()))
images, labels = batch['image'][0][:9], batch['label'][0][:9]
titles = map(make_label_getter(dataset), labels.argmax(axis=1))
show_img_grid(images, titles)


# ### Load pre-trained

# In[15]:


model_config = models_config.MODEL_CONFIGS[model_name]
model_config


# In[16]:


# Load model definition & initialize random parameters.
# This also compiles the model to XLA (takes some minutes the first time).
if model_name.startswith('Mixer'):
  model = models.MlpMixer(num_classes=num_classes, **model_config)
else:
  model = models.VisionTransformer(num_classes=num_classes, **model_config)
variables = jax.jit(lambda: model.init(
    jax.random.PRNGKey(0),
    # Discard the "num_local_devices" dimension of the batch for initialization.
    batch['image'][0, :1],
    train=False,
), backend='cpu')()


# In[17]:


# Load and convert pretrained checkpoint.
# This involves loading the actual pre-trained model results, but then also also
# modifying the parameters a bit, e.g. changing the final layers, and resizing
# the positional embeddings.
# For details, refer to the code and to the methods of the paper.
params = checkpoint.load_pretrained(
    pretrained_path=f'{model_name}.npz',
    init_params=variables['params'],
    model_config=model_config,
)


# ### Evaluate

# In[18]:


# So far, all our data is in the host memory. Let's now replicate the arrays
# into the devices.
# This will make every array in the pytree params become a ShardedDeviceArray
# that has the same data replicated across all local devices.
# For TPU it replicates the params in every core.
# For a single GPU this simply moves the data onto the device.
# For CPU it simply creates a copy.
params_repl = flax.jax_utils.replicate(params)
print('params.cls:', type(params['head']['bias']).__name__,
      params['head']['bias'].shape)
print('params_repl.cls:', type(params_repl['head']['bias']).__name__,
      params_repl['head']['bias'].shape)


# In[19]:


# Then map the call to our model's forward pass onto all available devices.
vit_apply_repl = jax.pmap(lambda params, inputs: model.apply(
    dict(params=params), inputs, train=False))


# In[20]:


def get_accuracy(params_repl):
  """Returns accuracy evaluated on the test set."""
  good = total = 0
  steps = input_pipeline.get_dataset_info(dataset, 'test')['num_examples'] // batch_size
  for _, batch in zip(tqdm.trange(steps), ds_test.as_numpy_iterator()):
    predicted = vit_apply_repl(params_repl, batch['image'])
    is_same = predicted.argmax(axis=-1) == batch['label'].argmax(axis=-1)
    good += is_same.sum()
    total += len(is_same.flatten())
  return good / total


# In[21]:


# Random performance without fine-tuning.
get_accuracy(params_repl)


# ### Fine-tune

# In[27]:


# experiment params
num_train_trials = 5
num_train_warmups = 1

# 100 Steps take approximately 15 minutes in the TPU runtime.
total_steps = 100
warmup_steps = 5
decay_type = 'cosine'
grad_norm_clip = 1
# This controls in how many forward passes the batch is split. 8 works well with
# a TPU runtime that has 8 devices. 64 should work on a GPU. You can of course
# also adjust the batch_size above, but that would require you to adjust the
# learning rate accordingly.
accum_steps = 8
base_lr = 0.03


# In[28]:


# Check out train.make_update_fn in the editor on the right side for details.
lr_fn = utils.create_learning_rate_schedule(total_steps, base_lr, decay_type, warmup_steps)
# We use a momentum optimizer that uses half precision for state to save
# memory. It als implements the gradient clipping.
tx = optax.chain(
    optax.clip_by_global_norm(grad_norm_clip),
    optax.sgd(
        learning_rate=lr_fn,
        momentum=0.9,
        accumulator_dtype='bfloat16',
    ),
)
update_fn_repl = train.make_update_fn(
    apply_fn=model.apply, accum_steps=accum_steps, tx=tx)
opt_state = tx.init(params)
opt_state_repl = flax.jax_utils.replicate(opt_state)


# In[29]:


# Initialize PRNGs for dropout.
update_rng_repl = flax.jax_utils.replicate(jax.random.PRNGKey(0))


# In[30]:


losses = []
lrs = []

import time
# run and time with the experiment params
track = 0
times = []
for run in tqdm.trange(1, num_train_trials + 1):
    start = time.time()
    
    # Completes in ~20 min on the TPU runtime.
    for step, batch in zip(
        tqdm.trange(1, total_steps + 1),
        ds_train.as_numpy_iterator(),
    ):
        params_repl, opt_state_repl, loss_repl, update_rng_repl = update_fn_repl(
        params_repl, opt_state_repl, batch, update_rng_repl)
        losses.append(loss_repl[0])
        lrs.append(lr_fn(step))

    end = time.time()
    time_run = end - start
    if track > num_train_warmups:
        times.append(time_run)

    track += 1

# get the average time taken
average_training_time = sum(times) / len(times)
print(times)
print(f"Average time to train {total_steps} steps: {average_training_time}")


# In[31]:


plt.plot(losses)
plt.figure()
plt.plot(lrs)


# In[32]:


# Should be ~96.7% for Mixer-B/16 or 97.7% for ViT-B/32 on CIFAR10 (both @224)
get_accuracy(params_repl)


# ### Inference

# In[33]:


# Download a pre-trained model.

if model_name.startswith('Mixer'):
  # Download model trained on imagenet2012
  get_ipython().system('[ -e "$model_name"_imagenet2012.npz ] || gsutil cp gs://mixer_models/imagenet1k/"$model_name".npz "$model_name"_imagenet2012.npz')
  model = models.MlpMixer(num_classes=1000, **model_config)
else:
  # Download model pre-trained on imagenet21k and fine-tuned on imagenet2012.
  get_ipython().system('[ -e "$model_name"_imagenet2012.npz ] || gsutil cp gs://vit_models/imagenet21k+imagenet2012/"$model_name".npz "$model_name"_imagenet2012.npz')
  model = models.VisionTransformer(num_classes=1000, **model_config)

import os
assert os.path.exists(f'{model_name}_imagenet2012.npz')


# In[34]:


# Load and convert pretrained checkpoint.
params = checkpoint.load(f'{model_name}_imagenet2012.npz')
params['pre_logits'] = {}  # Need to restore empty leaf for Flax.


# In[35]:


# Get imagenet labels.
get_ipython().system('wget https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt')
imagenet_labels = dict(enumerate(open('ilsvrc2012_wordnet_lemmas.txt')))


# In[36]:


# Get a random picture with the correct dimensions.
resolution = 224 if model_name.startswith('Mixer') else 384
get_ipython().system('wget https://picsum.photos/$resolution -O picsum.jpg')
import PIL
img = PIL.Image.open('picsum.jpg')
img


# In[37]:


# Predict on a batch with a single item (note very efficient TPU usage...)
logits, = model.apply(dict(params=params), (np.array(img) / 128 - 1)[None, ...], train=False)


# In[44]:


preds = np.array(jax.nn.softmax(logits))
for idx in preds.argsort()[:-11:-1]:
  print(f'{preds[idx]:.5f} : {imagenet_labels[idx]}', end='')


# In[ ]:


num_inference_trials = 100
num_inference_warmups = 10

import os

def filecount(dir: str) -> int:
    file_count = 0
    for entry in os.scandir(dir):
        if entry.is_file():
            file_count += 1
    return file_count

if !os.path.isdir("inference") and !(filecount("inference") >= num_inference_trials):
    resolution = 224 if model_name.startswith('Mixer') else 384
    get_ipython().system('mkdir -p inference')
    for index in range(1, num_inference_trials + 1):
        output = f"picsum{index}.jpg"
        get_ipython().system('wget https://picsum.photos/$resolution -O inference/$output')


# In[49]:


track = 0
inference_times = []
for i in tqdm.trange(1, num_inference_trials + 1):
    img = PIL.Image.open(f"inference/picsum{i}.jpg")
    start_inference = time.time()
    
    # note that this is not saturating the GPU, a larger batch size would be better
    logits, = model.apply(dict(params=params), (np.array(img) / 128 - 1)[None, ...], train=False)

    end_inference = time.time()
    if track > num_inference_warmups:
        inference_times.append(end_inference - start_inference)
    track += 1

average_inference_time = sum(inference_times) / len(inference_times)
print(inference_times)
print(f"Average inference time: {average_inference_time}")


# In[46]:


# export the python file
get_ipython().system('jupyter nbconvert --to script vit_jax.ipynb')
get_ipython().system('mv vit_jax.py pvit_jax.py')

print(f"Average Training Time: {average_training_time}")
print(f"Average Inference Time: {average_inference_time}")
print(f"Average JIT Time: TODO")


# In[ ]:




