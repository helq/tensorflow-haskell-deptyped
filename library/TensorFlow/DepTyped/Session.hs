{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}

module TensorFlow.DepTyped.Session (
  Runnable(runWithFeeds, run)
) where

import qualified Data.Vector as VN (Vector)
import           Data.Vector.Sized (Vector, toSized)
import           Data.Maybe (fromMaybe)

import qualified TensorFlow.Tensor as TF (Feed)
import qualified TensorFlow.Session as TF (runWithFeeds, run, SessionT)
import qualified TensorFlow.Types as TF (TensorDataType)
import           Control.Monad.IO.Class (MonadIO)
import qualified Foreign.Storable as FS (Storable)

import           GHC.TypeLits (KnownNat, Nat, Symbol)

import           TensorFlow.DepTyped.Tensor (Tensor(Tensor), FeedList(NilFeedList,(:~~)), Feed(Feed), SortPlaceholderList)
import           TensorFlow.DepTyped.Output (ControlNode(ControlNode))
import           TensorFlow.DepTyped.Base (ShapeProduct)

class Runnable (flphs :: [(Symbol, [Nat])]) tt rs a | tt -> a, rs -> a where
  runWithFeeds :: MonadIO m => FeedList flphs a -> tt -> TF.SessionT m rs
  run :: (MonadIO m, flphs ~ '[]) => tt -> TF.SessionT m rs

instance (FS.Storable a,
          TF.TensorDataType VN.Vector a,
          SortPlaceholderList phs1 ~ phs2)
       => Runnable phs1 (Tensor shape phs2 v a) (VN.Vector a) a where
  runWithFeeds feeds (Tensor t) = TF.runWithFeeds (getListFeeds feeds) t
  run (Tensor t) = TF.run t

instance SortPlaceholderList phs1 ~ phs2 => Runnable phs1 (ControlNode phs2) () () where
  runWithFeeds feeds (ControlNode cn) = TF.runWithFeeds (getListFeeds feeds) cn
  run (ControlNode cn) = TF.run cn

-- Sized Version of runWithFeeds output
-- TODO(helq): remove error that appears in fromMaybe
instance (FS.Storable a,
          TF.TensorDataType VN.Vector a,
          KnownNat n,
          ShapeProduct shape ~ n,
          SortPlaceholderList phs1 ~ phs2)
       => Runnable phs1 (Tensor shape phs2 v a) (Vector n a) a where
  runWithFeeds feeds (Tensor t) = fromMaybe (error "possible size mismatch between output vector and tensor shape, this should never happen :S")
                                  . toSized <$> TF.runWithFeeds (getListFeeds feeds) t
  run (Tensor t) = fromMaybe (error "possible size mismatch between output vector and tensor shape, this should never happen :S")
                     . toSized <$> TF.run t

getListFeeds :: FeedList phs a -> [TF.Feed]
getListFeeds NilFeedList     = []
getListFeeds (Feed f :~~ fs) = f : getListFeeds fs
