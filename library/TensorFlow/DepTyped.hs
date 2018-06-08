-- Copyright 2017 Elkin Cruz.
-- Copyright 2017 James Bowen.
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

module TensorFlow.DepTyped (
  KnownNatList(natListVal),
  constant, placeholder, add, mul, matMul, argMax, softmax, scalar, oneHot,
  oneHot_, reduceMean, softmaxCrossEntropyWithLogits, equal, truncatedNormal,
  relu, sub, cast, square, reshape, shape, sigmoid,
  run, runWithFeeds,
  Tensor, Placeholder, Feed(Feed), FeedList(NilFeedList,(:~~)), render, feed,
  TensorData(TensorData, unTensorData), encodeTensorData,
  Variable(Variable, unVariable), initializedVariable, zeroInitializedVariable, readValue,
  minimizeWith,
  sigmoidCrossEntropyWithLogits,

  Build, Value, Ref, MonadBuild,
  Session, runSession
) where

import TensorFlow.DepTyped.Ops (constant, placeholder, add, mul, matMul, argMax, softmax, scalar, oneHot,
                                oneHot_, reduceMean, softmaxCrossEntropyWithLogits, equal, truncatedNormal,
                                relu, sub, cast, square, reshape, shape, sigmoid)
import TensorFlow.DepTyped.Session (run, runWithFeeds)
import TensorFlow.DepTyped.Tensor (Tensor, Placeholder, Feed(Feed), FeedList(NilFeedList,(:~~)), render, feed)
import TensorFlow.DepTyped.Types (TensorData(TensorData, unTensorData), encodeTensorData)
import TensorFlow.DepTyped.Base (KnownNatList(natListVal))
import TensorFlow.DepTyped.Variable (Variable(Variable, unVariable), initializedVariable, zeroInitializedVariable, readValue)
import TensorFlow.DepTyped.Minimize (minimizeWith)
import TensorFlow.DepTyped.NN (sigmoidCrossEntropyWithLogits)

import TensorFlow.Core (Build, Value, Ref, MonadBuild)
import TensorFlow.Session (Session, runSession)
