"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Operations used by Grain"""

import dataclasses
import warnings
from typing import Dict
from threading import current_thread
import datasets
from datasets.distributed import split_dataset_by_node
import numpy as np
import tensorflow as tf
from .. import max_logging
from .. import tokenizer

Features = Dict[str, tf.Tensor]
AUTOTUNE = tf.data.experimental.AUTOTUNE

########## Functions used by TFDS pipeline


def normalize_features(x, column_name):
  return {"inputs": x[column_name], "targets": x[column_name]}


def get_tokenizer(tokenizer_path, add_bos, add_eos):
  # Load tokenizer
  tokenizer_model = tokenizer.build_tokenizer(tokenizer_path, add_bos, add_eos)
  return tokenizer_model


def truncate_to_max_allowable_length(x, max_length):
  x["inputs"] = x["inputs"][:max_length]
  x["targets"] = x["targets"][:max_length]
  return x


def shift_data_by_truncation(x):
  x["inputs"] = x["inputs"][:-1]
  x["targets"] = x["targets"][1:]
  return x


########## Functions used by HF pipeline


def tokenization(example, hf_tokenizer, max_length, column_name):
  """Tokenize a HuggingFace dataset"""
  return hf_tokenizer(example[column_name], truncation=True, max_length=max_length)


def shift_right(x, axis=1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  slices = [
      slice(None),
  ] * len(x.shape)
  slices[axis] = slice(0, -1)
  padded = np.pad(x, pad_widths, mode="constant", constant_values=x.dtype.type(0))
  return padded[tuple(slices)]


def shift_and_refine(x, axis=1):
  """Shift inputs, set segmentation to 0 when target element is 0.
  Replace EOS by 0 for packed inputs."""
  x["inputs"] = shift_right(x["inputs"], axis=axis)
  targets_nonzero = x["targets"] != 0
  x["inputs_segmentation"] *= targets_nonzero
  x["targets_segmentation"] *= targets_nonzero
  # For packed targets, the first shifted token of a new sequence is made
  # 0, rather than being the EOS token for the last sequence.
  x["inputs"] *= x["inputs_segmentation"] == shift_right(x["inputs_segmentation"], axis=axis)

  return x