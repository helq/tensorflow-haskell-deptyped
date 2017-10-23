module TensorFlow.DepTyped (
  constant, placeholder, add, matMul,
  run, runWithFeeds,
  Tensor(Tensor), Placeholder, Feed(Feed), FeedList(NilFeedList,(:~~)), render, feed,
  TensorData(TensorData), encodeTensorData,

  Build, Value, Ref, MonadBuild,
  Session, runSession
) where

import TensorFlow.DepTyped.Ops (constant, placeholder, add, matMul)
import TensorFlow.DepTyped.Session (run, runWithFeeds)
import TensorFlow.DepTyped.Tensor (Tensor(Tensor), Placeholder, Feed(Feed), FeedList(NilFeedList,(:~~)), render, feed)
import TensorFlow.DepTyped.Types (TensorData(TensorData), encodeTensorData)

import TensorFlow.Core (Build, Value, Ref, MonadBuild)
import TensorFlow.Session (Session, runSession)
