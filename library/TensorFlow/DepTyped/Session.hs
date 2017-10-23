{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}

module TensorFlow.DepTyped.Session (
  run,
  runWithFeeds
) where

import           TensorFlow.Nodes (Fetchable)
import qualified TensorFlow.Tensor as TF (Tensor, Feed)
import qualified TensorFlow.Session as TF (runWithFeeds, run, Session)

import           TensorFlow.DepTyped.Tensor (Tensor(Tensor), FeedList(NilFeedList,(:~~)), Feed(Feed), SortPlaceholderList)

-- TODO(helq): look in how to convert the result from a VN.Vector to a normal vector
runWithFeeds :: (Fetchable (TF.Tensor v a) result, SortPlaceholderList phs1 ~ phs2)
             => FeedList phs1 a -> Tensor shape phs2 v a -> TF.Session result
runWithFeeds feeds (Tensor t) = TF.runWithFeeds (getListFeeds feeds) t
  where getListFeeds :: FeedList phs a -> [TF.Feed]
        getListFeeds NilFeedList     = []
        getListFeeds (Feed f :~~ fs) = f : getListFeeds fs

run :: (Fetchable (TF.Tensor v a) result)
    => Tensor shape '[] v a -> TF.Session result
run (Tensor t) = TF.run t
