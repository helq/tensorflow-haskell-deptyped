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

{-# LANGUAGE DataKinds              #-}
{-# LANGUAGE FlexibleContexts       #-}
{-# LANGUAGE TypeFamilies           #-}
{-# LANGUAGE MultiParamTypeClasses  #-}
{-# LANGUAGE ScopedTypeVariables    #-}
{-# LANGUAGE TypeSynonymInstances   #-}
{-# LANGUAGE FlexibleInstances      #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE TypeInType             #-}

module TensorFlow.DepTyped.Session (
  Runnable(runWithFeeds, run)
) where

import qualified Data.Vector as VN (Vector)
import           Data.Vector.Sized (Vector, toSized)
import           Data.Maybe (fromMaybe)

import qualified TensorFlow.Tensor as TF (Feed)
import qualified TensorFlow.Session as TF (runWithFeeds, run, SessionT)
import qualified TensorFlow.Types as TF (TensorDataType, Scalar)
import           Control.Monad.IO.Class (MonadIO)
import qualified Foreign.Storable as FS (Storable)

import           GHC.TypeLits (KnownNat, Nat, Symbol)
import           Data.Kind (Type)

import           TensorFlow.DepTyped.Tensor (Tensor(Tensor), FeedList(NilFeedList,(:~~)), Feed(Feed))
import           TensorFlow.DepTyped.Output (ControlNode(ControlNode))
import           TensorFlow.DepTyped.Base (ShapeProduct, SortPlaceholderList)

-- TODO(helq): find a way to created "dependent typed" version of run_ and runWithFeeds_

-- TODO(helq): add instances for (,), (,,), so on
class Runnable (feedlist_phs :: [(Symbol, [Nat], Type)]) tt rs a | tt -> a, rs -> a where
  runWithFeeds :: MonadIO m => FeedList feedlist_phs -> tt -> TF.SessionT m rs
  run :: (MonadIO m, feedlist_phs ~ '[]) => tt -> TF.SessionT m rs

instance (FS.Storable a,
          TF.TensorDataType VN.Vector a,
          SortPlaceholderList feedlist_phs ~ phs)
       => Runnable feedlist_phs (Tensor shape phs v a) (VN.Vector a) a where
  runWithFeeds feeds (Tensor t) = TF.runWithFeeds (getListFeeds feeds) t
  run (Tensor t) = TF.run t

instance SortPlaceholderList feedlist_phs ~ phs => Runnable feedlist_phs (ControlNode phs) () () where
  runWithFeeds feeds (ControlNode cn) = TF.runWithFeeds (getListFeeds feeds) cn
  run (ControlNode cn) = TF.run cn

-- Sized Version of runWithFeeds output
-- TODO(helq): remove error that appears in fromMaybe
instance (FS.Storable a,
          TF.TensorDataType VN.Vector a,
          KnownNat n,
          ShapeProduct shape ~ n,
          SortPlaceholderList feedlist_phs ~ phs)
       => Runnable feedlist_phs (Tensor shape phs v a) (Vector n a) a where
  runWithFeeds feeds (Tensor t) = fromMaybe (error "possible size mismatch between output vector and tensor shape, this should never happen :S")
                                  . toSized <$> TF.runWithFeeds (getListFeeds feeds) t
  run (Tensor t) = fromMaybe (error "possible size mismatch between output vector and tensor shape, this should never happen :S")
                     . toSized <$> TF.run t

instance (FS.Storable a,
          TF.TensorDataType VN.Vector a,
          SortPlaceholderList feedlist_phs ~ phs)
       => Runnable feedlist_phs (Tensor '[1] phs v a) (TF.Scalar a) a where
  runWithFeeds feeds (Tensor t) = TF.runWithFeeds (getListFeeds feeds) t
  run (Tensor t) = TF.run t

getListFeeds :: FeedList phs -> [TF.Feed]
getListFeeds NilFeedList     = []
getListFeeds (Feed f :~~ fs) = f : getListFeeds fs
