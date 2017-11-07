module TensorFlow.DepTyped (
  KnownNatList(natListVal),
  constant, placeholder, add, mul, matMul, argMax, softmax, scalar, oneHot,
  oneHot_, reduceMean, softmaxCrossEntropyWithLogits, equal, truncatedNormal,
  relu, sub, cast,
  run, runWithFeeds,
  Tensor, Placeholder, Feed(Feed), FeedList(NilFeedList,(:~~)), render, feed,
  TensorData(TensorData, unTensorData), encodeTensorData,
  Variable(Variable, unVariable), initializedVariable, zeroInitializedVariable, readValue,
  minimizeWith,

  Build, Value, Ref, MonadBuild,
  Session, runSession
) where

import TensorFlow.DepTyped.Ops (constant, placeholder, add, mul, matMul, argMax, softmax, scalar, oneHot,
                                oneHot_, reduceMean, softmaxCrossEntropyWithLogits, equal, truncatedNormal,
                                relu, sub, cast)
import TensorFlow.DepTyped.Session (run, runWithFeeds)
import TensorFlow.DepTyped.Tensor (Tensor, Placeholder, Feed(Feed), FeedList(NilFeedList,(:~~)), render, feed)
import TensorFlow.DepTyped.Types (TensorData(TensorData, unTensorData), encodeTensorData)
import TensorFlow.DepTyped.Base (KnownNatList(natListVal))
import TensorFlow.DepTyped.Variable (Variable(Variable, unVariable), initializedVariable, zeroInitializedVariable, readValue)
import TensorFlow.DepTyped.Minimize (minimizeWith)

import TensorFlow.Core (Build, Value, Ref, MonadBuild)
import TensorFlow.Session (Session, runSession)
