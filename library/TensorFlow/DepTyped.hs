module TensorFlow.DepTyped (
  KnownNatList(natListVal),
  constant, placeholder, add, matMul,
  run, runWithFeeds,
  Tensor(Tensor), Placeholder, Feed(Feed), FeedList(NilFeedList,(:~~)), render, feed,
  TensorData(TensorData, unTensorData), encodeTensorData,
  Variable(Variable, unVariable), initializedVariable, zeroInitializedVariable,

  Build, Value, Ref, MonadBuild,
  Session, runSession
) where

import TensorFlow.DepTyped.Ops (constant, placeholder, add, matMul)
import TensorFlow.DepTyped.Session (run, runWithFeeds)
import TensorFlow.DepTyped.Tensor (Tensor(Tensor), Placeholder, Feed(Feed), FeedList(NilFeedList,(:~~)), render, feed)
import TensorFlow.DepTyped.Types (TensorData(TensorData, unTensorData), encodeTensorData)
import TensorFlow.DepTyped.Base (KnownNatList(natListVal))
import TensorFlow.DepTyped.Variable (Variable(Variable, unVariable), initializedVariable, zeroInitializedVariable)

import TensorFlow.Core (Build, Value, Ref, MonadBuild)
import TensorFlow.Session (Session, runSession)
